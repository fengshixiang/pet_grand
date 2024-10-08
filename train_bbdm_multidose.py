import pandas as pd
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
add_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(add_dir)
import re
import os.path as osp
import SimpleITK as sitk
import torch.distributed as dist

from utils.dataset import LQDataset_Multidose, dynamic_resize_collate
from utils.loggerx import LoggerX
from utils.sampler import RandomSampler, TestSampler
from utils.utils import MorphologicalOperation, GetLargestConnectedCompont, GetLargestConnectedCompontBoundingbox
import argparse
import tqdm
import copy
import torch
import numpy as np
from utils.ema import EMA
import torchvision
from utils.measure import compute_measure, compute_measure_single
from utils.evaluation import compute_nrmse, compute_psnr, compute_ssim
from models.bbdm.diffusion_network import Network
from models.bbdm.diffusion_process import Diffusion
import traceback
import cv2

torch.manual_seed(332)
device = torch.device("cuda")

parser = argparse.ArgumentParser('Default arguments for training of different methods')

parser.add_argument('--save_freq', type=int, default=2,  # 2500
                    help='save frequency of images and models')
parser.add_argument('--metric_freq', type=int, default=2,  # save_freq greater than metric_freq
                    help='metric frequency')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test_batch_size')
parser.add_argument('--num_workers', type=int, default=1,
                    help='num of workers to use')
parser.add_argument('--max_iter', type=int, default=150000,
                    help='number of training iterations')
parser.add_argument('--resume_iter', type=int, default=0,
                    help='number of training epochs')
parser.add_argument('--test_iter', type=int, default=0,
                    help='number of epochs for test')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument('--port', type=str, default='23485')


# run_name and model_name
parser.add_argument('--run_name', type=str, default='random_4_T20',
                    help='each run name')
parser.add_argument('--model_name', type=str, default='bbdm',
                    help='the type of method')

parser.add_argument('--objective', type=str, default='grad',
                    help='the type of method')
parser.add_argument('--beta_schedule', type=str, default='sigmoid',
                    help='the type of method')
# dataset
parser.add_argument('--mod_input', type=str, default='random_4', help='input modality')
parser.add_argument('--mod_target', type=str, default='target', help='target modality')
parser.add_argument('--train_dataset', type=str, default='01')
parser.add_argument('--test_dataset', type=str, default='01')
parser.add_argument('--data_root', type=str, default='/data/songtao/datas/generation/fastMRI_Brain_npy', help='train data path')
parser.add_argument('--context', default=False, action="store_true", help='use contextual information')
parser.add_argument('--mask', default=False, action="store_true", help='use mask roi')
parser.add_argument('--augment', default=False, action="store_true", help='use brush mask augmentation') 


## diffusion traning
parser.add_argument("--in_channels", default=2, type=int)
parser.add_argument("--out_channels", default=1, type=int)
parser.add_argument("--init_lr", default=2e-4, type=float)

parser.add_argument('--update_ema_iter', default=8, type=int)
parser.add_argument('--start_ema_iter', default=30000, type=int)
parser.add_argument('--ema_decay', default=0.9999, type=float)

parser.add_argument('--T', default=20, type=int)

parser.add_argument('--sampling_timesteps', default=20, type=int)
parser.add_argument('--loss_type', default='L2', type=str)

parser.add_argument('--max_var', default=1, type=float)

parser.add_argument('--size', default=256, type=int)
parser.add_argument('--skip_sample', default=False, action="store_true", help='use mask roi')
parser.add_argument('--normalize_type', default=1, type=int)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def normalized_to_zero_to_one(t):
    t = (t - t.min()) / (t.max() - t.min())

    return t

class Trainer(object):

    def __init__(self, opt):
        self.opt = opt
        self.base_size = 32
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.logger = LoggerX(save_root=osp.join(osp.dirname(osp.abspath(__file__)), 'output', '{}_{}'.format(opt.model_name, opt.run_name)),
                              local_rank=self.rank, samplings_step=opt.sampling_timesteps, mode=opt.mode)

        if opt.mode == 'train':
            self.set_loader()
        self.set_model()

    def set_loader(self):
        opt = self.opt
        if opt.mode == 'train':
            train_ds = LQDataset_Multidose(mode='train', data_root=opt.data_root,
                                           mod_target=opt.mod_target, context=opt.context,
                                           mask=opt.mask, augment=opt.augment, normalize_type=opt.normalize_type)
            train_sampler = RandomSampler(dataset=train_ds, batch_size=opt.batch_size,
                                          num_iter=opt.max_iter,
                                          restore_iter=opt.resume_iter)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=opt.batch_size,
                sampler=train_sampler,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=True,
                collate_fn=dynamic_resize_collate
            )
            self.train_loader = train_loader

        test_ds = LQDataset_Multidose(mode='test', data_root=opt.data_root,
                                      mod_target=opt.mod_target, context=opt.context, mask=opt.mask,
                                      normalize_type=opt.normalize_type)
        test_sampler = TestSampler(dataset=test_ds, batch_size=opt.test_batch_size)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=opt.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=opt.num_workers,
            pin_memory=True,
            collate_fn=dynamic_resize_collate
        )
        self.test_loader = test_loader
        self.length = test_ds.__len__()

        test_images = dynamic_resize_collate([test_ds[0]])
        low_dose = test_images['input'].to(device)
        full_dose = test_images['target'].to(device)
        if opt.mask:
            mask = torch.stack([torch.from_numpy(x['mask']) for x in test_images], dim=0).to(device)
            self.test_images = (low_dose, full_dose, mask)
        else:
            self.test_images = (low_dose, full_dose)

        self.test_dataset = test_ds


    def fit(self):
        opt = self.opt
        if self.rank == 0:
            self.logger.text_logger.info('save_freq:{}'.format(opt.save_freq))
            self.logger.text_logger.info('batch_size:{}'.format(opt.batch_size))
            self.logger.text_logger.info('init_lr:{}'.format(opt.init_lr))
            self.logger.text_logger.info('in_channels:{}'.format(opt.in_channels))
            self.logger.text_logger.info('out_channels:{}'.format(opt.out_channels))
            self.logger.text_logger.info('T:{}'.format(opt.T))
            self.logger.text_logger.info('mask:{}'.format(opt.mask))
            self.logger.text_logger.info('augment:{}'.format(opt.augment))
            self.logger.text_logger.info('loss_type:{}'.format(opt.loss_type))
            self.logger.text_logger.info('sampling_timesteps:{}'.format(opt.sampling_timesteps))
            self.logger.text_logger.info('start_ema_iter:{}'.format(opt.start_ema_iter))
            self.logger.text_logger.info('update_ema_iter:{}'.format(opt.update_ema_iter))
            self.logger.text_logger.info('ema_decay:{}'.format(opt.ema_decay))
            self.logger.text_logger.info('train_dataset:{}'.format(opt.train_dataset))
            self.logger.text_logger.info('test_dataset:{}'.format(opt.test_dataset))
            self.logger.text_logger.info('data_root:{}'.format(opt.data_root))
            self.logger.text_logger.info('mod_input:{}'.format(opt.mod_input))
            self.logger.text_logger.info('mod_target:{}'.format(opt.mod_target))
            self.logger.text_logger.info('context:{}'.format(opt.context))
            self.logger.text_logger.info('num_workers:{}'.format(opt.num_workers))
            self.logger.text_logger.info('max_iter:{}'.format(opt.max_iter))
            self.logger.text_logger.info('resume_iter:{}'.format(opt.resume_iter))
            self.logger.text_logger.info('test_iter:{}'.format(opt.test_iter))
            self.logger.text_logger.info('objective:{}'.format(opt.objective))
            self.logger.text_logger.info('beta_schedule:{}'.format(opt.beta_schedule))
            self.logger.text_logger.info('normalize_type:{}'.format(opt.normalize_type))

        if opt.mode == 'train':
            if opt.resume_iter > 0:
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        self.logger.load_checkpoints(opt.resume_iter)
                else:
                    self.logger.load_checkpoints(opt.resume_iter)
                if dist.is_initialized():
                    dist.barrier()
            if dist.is_initialized():
                broadcast_params(self.model.parameters())
                broadcast_params(self.ema_model.parameters())
            # training routine
            loader = iter(self.train_loader)
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                inputs = next(loader)
                self.train(inputs, n_iter)
                if self.rank == 0:
                    if n_iter % opt.save_freq == 0:
                        self.logger.checkpoints(n_iter)
                        self.generate_images(n_iter)
                if n_iter % opt.metric_freq == 0:
                    self.test(n_iter)
                    
                if dist.is_initialized():
                    dist.barrier()

        elif opt.mode == 'test':
            if opt.test_iter > 0:
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        self.logger.load_checkpoints(opt.test_iter)
                else:
                    self.logger.load_checkpoints(opt.test_iter)
                if dist.is_initialized():
                    dist.barrier()
            if dist.is_initialized():
                broadcast_params(self.model.parameters())
                broadcast_params(self.ema_model.parameters())
            
            self.test(opt.test_iter)
            if dist.is_initialized():
                dist.barrier()
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    self.generate_images(opt.test_iter)
            else:
                self.generate_images(opt.test_iter)

        elif opt.mode == 'test_patient':
            if opt.test_iter > 0:
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        self.logger.load_checkpoints(opt.test_iter)
                else:
                    self.logger.load_checkpoints(opt.test_iter)
                if dist.is_initialized():
                    dist.barrier()
            if dist.is_initialized():
                broadcast_params(self.model.parameters())
                broadcast_params(self.ema_model.parameters())

            self.test_patient(opt.test_iter)
            if dist.is_initialized():
                dist.barrier()
        elif opt.mode == 'test_patient_grand_challenge':
            if opt.test_iter > 0:
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        self.logger.load_checkpoints(opt.test_iter)
                else:
                    self.logger.load_checkpoints(opt.test_iter)
                if dist.is_initialized():
                    dist.barrier()
            if dist.is_initialized():
                broadcast_params(self.model.parameters())
                broadcast_params(self.ema_model.parameters())

            self.test_patient_grand_challenge(opt.test_iter)
            if dist.is_initialized():
                dist.barrier()


    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.T = opt.T
        self.sampling_timesteps = opt.sampling_timesteps
        self.context = opt.context
        denoise_fn = Network(in_channels=opt.in_channels, out_channels=opt.out_channels, context=opt.context)
        if self.rank == 0:
            self.logger.text_logger.info(denoise_fn)
        model = Diffusion(
            denoise_fn=denoise_fn,
            objective=opt.objective,
            timesteps=opt.T,
            sample_step=opt.sampling_timesteps,
            max_var=opt.max_var,
            skip_sample=opt.skip_sample
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), opt.init_lr, weight_decay=0)
        if dist.is_initialized():
            rank = dist.get_rank()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank, ],output_device=rank, find_unused_parameters=True)
        
        ema_model = copy.deepcopy(model)
        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model


        self.reset_parameters()

    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        ## training process
        if opt.mask:
            low_dose, full_dose, mask = inputs['input'].to(device), inputs['target'].to(device), inputs['mask'].to(device)
        else:
            low_dose, full_dose, mask = inputs['input'].to(device), inputs['target'].to(device), None
        loss, _ = self.model(full_dose, low_dose, mask)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        if self.rank == 0:
            self.logger.msg([loss, lr], n_iter)

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)

    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        psnr = torch.tensor(0, dtype=torch.float32, device=device)
        ssim = torch.tensor(0, dtype=torch.float32, device=device)
        nmse = torch.tensor(0, dtype=torch.float32, device=device)
        psnr_o = torch.tensor(0, dtype=torch.float32, device=device)
        ssim_o = torch.tensor(0, dtype=torch.float32, device=device)
        nmse_o = torch.tensor(0, dtype=torch.float32, device=device)

        for inputs in tqdm.tqdm(self.test_loader, desc='test'):
            if opt.mask:
                low_dose, full_dose, mask = inputs['input'].to(device), inputs['target'].to(device), inputs['mask'].to(device)
            else:
                low_dose, full_dose, mask = inputs['input'].to(device), inputs['target'].to(device), None

            if dist.is_initialized():
                gen_full_dose = self.ema_model.module.sample(low_dose, mask)
            else:
                gen_full_dose = self.ema_model.sample(low_dose, mask)
            if opt.context:
                low_dose = low_dose[:, 1:2]
                full_dose = full_dose[:, 1:2]
                gen_full_dose = gen_full_dose[:, 1:2]
            data_range = full_dose.max() - full_dose.min()
            psnr_score, ssim_score, nmse_score = compute_measure(full_dose, gen_full_dose, data_range, self.length)
            psnr_score_o, ssim_score_o, nmse_score_o = compute_measure(full_dose, low_dose, data_range, self.length)
            psnr += torch.tensor(psnr_score, dtype=torch.float32, device=device)
            ssim += torch.tensor(ssim_score, dtype=torch.float32, device=device)
            nmse += torch.tensor(nmse_score, dtype=torch.float32, device=device)
            psnr_o += torch.tensor(psnr_score_o, dtype=torch.float32, device=device)
            ssim_o += torch.tensor(ssim_score_o, dtype=torch.float32, device=device)
            nmse_o += torch.tensor(nmse_score_o, dtype=torch.float32, device=device)
        # merge results from different ranks
        if dist.is_initialized():
            dist.barrier()
            dist.reduce(psnr, dst=0)
            dist.reduce(nmse, dst=0)
            dist.reduce(ssim, dst=0)    
            dist.barrier()
            if dist.get_rank() == 0:
                self.logger.msg([psnr.item(), ssim.item(), nmse.item()], n_iter)
                self.logger.msg([psnr_o.item(), ssim_o.item(), nmse_o.item()], n_iter)
        else:
            self.logger.msg([psnr.item(), ssim.item(), nmse.item()], n_iter)
            self.logger.msg([psnr_o.item(), ssim_o.item(), nmse_o.item()], n_iter)

    @torch.no_grad()
    def test_patient_grand_challenge(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        itk_save_path = osp.join(osp.dirname(osp.abspath(__file__)), 'output', '{}_{}'.format(opt.model_name, opt.run_name),
                                 'save_itk_{}_grand_challenge'.format(opt.test_iter))
        itk_merge_save_path = itk_save_path + '_merge'
        os.makedirs(itk_save_path, exist_ok=True)
        os.makedirs(itk_merge_save_path, exist_ok=True)

        patient_p_list = subdirs(opt.data_root)

        for patient_p in patient_p_list[25:]:
            series_n_list = subfiles(patient_p)
            input_p = series_n_list[0]
            low_itk = sitk.ReadImage(input_p)

            ##### get roi
            minimumfilter = sitk.MinimumMaximumImageFilter()
            minimumfilter.Execute(low_itk)
            binary_src = sitk.BinaryThreshold(low_itk, 50, minimumfilter.GetMaximum())
            binary_src = MorphologicalOperation(binary_src, 5)
            binary_src = GetLargestConnectedCompont(binary_src)
            boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
            x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
            low_arr_origin = sitk.GetArrayFromImage(low_itk)
            # 根据boundingbox得到ROI区域
            roi_low_arr = low_arr_origin[z1:z2, y1:y2, x1:x2]

            ##### test
            low_arr = roi_low_arr.astype(np.float32)
            mean, std, perc995 = low_arr.mean(), low_arr.std(), np.percentile(low_arr, 99.5)
            if opt.normalize_type == 1:
                low_arr = np.clip(low_arr, 0, perc995) / perc995
                low_arr = normalize_to_neg_one_to_one(low_arr)
            elif opt.normalize_type == 2:
                low_arr = (low_arr - mean) / std
            elif opt.normalize_type == 3:
                low_arr = low_arr / mean
            elif opt.normalize_type == 4:
                low_arr = low_arr / (mean + 3 * std)
            else:
                raise ValueError
            h, w = low_arr.shape[1:]
            new_h = int(np.ceil(h / self.base_size)) * self.base_size
            new_w = int(np.ceil(w / self.base_size)) * self.base_size
            low_arr_padded = low_arr
            # 填充
            if h < new_h:
                pad_h = (new_h - h) // 2
                low_arr_padded = np.pad(low_arr_padded, ((0, 0), (pad_h, new_h - h - pad_h), (0, 0)),
                                        mode='reflect')
            if w < new_w:
                pad_w = (new_w - w) // 2
                low_arr_padded = np.pad(low_arr_padded, ((0, 0), (0, 0), (pad_w, new_w - w - pad_w)),
                                        mode='reflect')
            gen_full_padded = np.zeros_like(low_arr_padded)
            gen_full_padded_merge = np.zeros_like(low_arr_padded)
            for i in range(low_arr_padded.shape[0]):
                if not opt.context:
                    input = low_arr_padded[i]
                    input = input.astype(np.float32)[np.newaxis, ...][np.newaxis, ...]
                else:
                    inputs, targets = [], []
                    for j in range(i - 1, i + 2):
                        j = min(low_arr_padded.shape[0] - 1, max(0, j))
                        input = low_arr_padded[j]
                        inputs.append(input)
                    inputs = np.stack(inputs)
                    input = inputs.astype(np.float32)[np.newaxis, ...]
                low_dose = torch.from_numpy(input).to(device)
                mask = None
                if dist.is_initialized():
                    # with autocast():
                    gen_full_dose = self.ema_model.module.sample(low_dose, mask)
                else:
                    # with autocast():
                    gen_full_dose = self.ema_model.sample(low_dose, mask)

                if not opt.context:
                    gen_full_padded[i] = gen_full_dose[0][0].cpu().numpy()
                else:
                    gen_full_padded[i] = gen_full_dose[0][1].cpu().numpy()
                    if 0 < i < low_arr_padded.shape[0] - 1:
                        gen_full_padded_merge[i - 1: i + 2] += gen_full_dose[0].cpu().numpy()
                    elif i == 0:
                        gen_full_padded_merge[:2] += gen_full_dose[0][1:].cpu().numpy()
                    else:
                        gen_full_padded_merge[-2:] += gen_full_dose[0][:2].cpu().numpy()

            gen_full_padded_merge[0] /= 2
            gen_full_padded_merge[-1] /= 2
            gen_full_padded_merge[1:-1] /= 3

            if h < new_h:
                gen_full_padded = gen_full_padded[:, pad_h: low_arr.shape[1] + pad_h]
                gen_full_padded_merge = gen_full_padded_merge[:, pad_h: low_arr.shape[1] + pad_h]
            if w < new_w:
                gen_full_padded = gen_full_padded[:, :, pad_w: low_arr.shape[2] + pad_w]
                gen_full_padded_merge = gen_full_padded_merge[:, :, pad_w: low_arr.shape[2] + pad_w]

            if opt.normalize_type == 1:
                gen_full_arr = unnormalize_to_zero_to_one(gen_full_padded)
                gen_full_arr_merge = unnormalize_to_zero_to_one(gen_full_padded_merge)
                gen_full_arr = gen_full_arr * perc995
                gen_full_arr_merge = gen_full_arr_merge * perc995
            elif opt.normalize_type == 2:
                gen_full_arr = gen_full_padded * std + mean
                gen_full_arr_merge = gen_full_padded_merge * std + mean
            elif opt.normalize_type == 3:
                gen_full_arr = gen_full_padded * mean
                gen_full_arr_merge = gen_full_padded_merge * mean
            elif opt.normalize_type == 4:
                gen_full_arr = gen_full_padded * (mean + 3 * std)
                gen_full_arr_merge = gen_full_padded_merge * (mean + 3 * std)
            else:
                raise ValueError
            gen_full_arr = np.clip(gen_full_arr, 0, gen_full_arr.max()).astype(np.float32)
            gen_full_arr_merge = np.clip(gen_full_arr_merge, 0, gen_full_arr_merge.max()).astype(np.float32)

            ##### pad roi
            gen_full_arr = np.pad(gen_full_arr, ((z1, low_arr_origin.shape[0] - z2), (y1, low_arr_origin.shape[1] - y2), (x1, low_arr_origin.shape[2] - x2)))
            gen_full_arr_merge = np.pad(gen_full_arr_merge, ((z1, low_arr_origin.shape[0] - z2), (y1, low_arr_origin.shape[1] - y2), (x1, low_arr_origin.shape[2] - x2)))

            # save_itk
            gen_full_img = sitk.GetImageFromArray(gen_full_arr)
            gen_full_img.CopyInformation(low_itk)
            patient_save_path = osp.join(itk_save_path, input_p.split('/')[-2])
            os.makedirs(patient_save_path, exist_ok=True)
            sitk.WriteImage(gen_full_img, osp.join(patient_save_path, input_p.split('/')[-1]))
            gen_full_img = sitk.GetImageFromArray(gen_full_arr_merge)
            gen_full_img.CopyInformation(low_itk)
            patient_save_path = osp.join(itk_merge_save_path, input_p.split('/')[-2])
            os.makedirs(patient_save_path, exist_ok=True)
            sitk.WriteImage(gen_full_img, osp.join(patient_save_path, input_p.split('/')[-1]))


    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        if opt.mask:
            low_dose, full_dose, mask = self.test_images
        else:
            low_dose, full_dose = self.test_images
            mask = None
        if dist.is_initialized():
            gen_full_dose = self.ema_model.module.sample(low_dose, mask)
        else:
            gen_full_dose = self.ema_model.sample(low_dose, mask)

        b, c, w, h = low_dose.size()

        psnrs, ssims = [],  []
        for i in range(low_dose.shape[0]):
            if opt.context:
                data_range = full_dose[i:i+1, 1:2].max() - full_dose[i:i+1, 1:2].min()
                psnr, ssim, nmse = compute_measure_single(full_dose[i:i+1, 1:2], gen_full_dose[i:i+1, 1:2], data_range)
            else:
                data_range = full_dose[i:i+1].max() - full_dose[i:i+1].min()
                psnr, ssim, nmse = compute_measure_single(full_dose[i:i+1], gen_full_dose[i:i+1], data_range)
            psnrs.append(psnr)
            ssims.append(ssim)
        if not opt.context:
            low_dose = normalized_to_zero_to_one(low_dose)
            full_dose = normalized_to_zero_to_one(full_dose)
            gen_full_dose = normalized_to_zero_to_one(gen_full_dose)
            fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose])
        else:
            low_dose = normalized_to_zero_to_one(low_dose[:, 1:2])
            full_dose = normalized_to_zero_to_one(full_dose[:, 1:2])
            gen_full_dose = normalized_to_zero_to_one(gen_full_dose[:, 1:2])
            fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose])
        if opt.context:
            fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, 1, w, h))
        else:
            fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
        grid_img = torchvision.utils.make_grid(fake_imgs, nrow=3)
        grid_img = (grid_img.cpu().numpy() * 255).round().astype(np.uint8)
        text_arr = np.zeros((grid_img.shape[0], grid_img.shape[1], w), dtype=np.uint8)
        for i in range(int(grid_img.shape[1]/ w)):
            # cv2.putText(text_arr[0], "psnr: {:.2f}".format(psnrs[i]), (50, 120+i*h), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)
            # cv2.putText(text_arr[0], "ssim: {:.2f}".format(ssims[i]), (50, 160+i*h), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)
            cv2.putText(text_arr[0], "psnr: {:.2f}".format(psnrs[i]), (50, 40+i * h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(text_arr[0], "ssim: {:.2f}".format(ssims[i]), (50, 100+i * h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        grid_img = np.concatenate([grid_img, text_arr], -1)
        self.logger.save_image(torch.from_numpy(grid_img)/255,
                               n_iter, 'test_{}'.format(self.sampling_timesteps) + '_' + opt.test_dataset)##convert to 0-255


    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass

    def transfer_calculate_window(self, img, MIN_B=0, MAX_B=255):
        img = torch.clamp(img, 0, 1)
        img = img * (MAX_B - MIN_B) + MIN_B
        return img

    def transfer_display_window(self, img, MIN_B=0, MAX_B=255):
        img = torch.clamp(img, 0, 1)
        img = img * (MAX_B - MIN_B) + MIN_B
        return img

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


if __name__ == '__main__':
    local_dist = False if '/mnt/lustre' in args.data_root else True
    try:
        backend="nccl"
        if local_dist:
            dist.init_process_group(backend, init_method='env://')
            torch.cuda.set_device(dist.get_rank())
        else:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            local_rank = int(os.environ['SLURM_LOCALID'])
            node_list = str(os.environ['SLURM_NODELIST'])   
            node_parts = re.findall('[0-9]+', node_list)
            host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])
            port = args.port
            init_method = 'tcp://{}:{}'.format(host_ip, port)
            dist.init_process_group(backend, init_method=init_method, world_size=world_size, rank=rank)
            print('global rank is {}, local_rank is {}, world_size is {}, ip is {}'.format(rank, local_rank, world_size, host_ip))
            torch.cuda.set_device(dist.get_rank())
    except Exception as e:
        traceback.print_exc()
        print('disable distribution')
    
    model = Trainer(args)
    model.fit()
    if dist.is_initialized():
        dist.destroy_process_group()