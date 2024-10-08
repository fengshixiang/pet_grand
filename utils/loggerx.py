import torch
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import torch.nn as nn
import inspect
from utils.ops import reduce_tensor, load_network
import logging
import cv2
import matplotlib.pyplot as plt

def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root, local_rank=0, samplings_step=1000, mode='train'):
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = 1
        self.local_rank = local_rank
        self.samplimgs_step = samplings_step
        if self.local_rank == 0:
            # set up logger
            logging.basicConfig()
            self.text_logger = logging.getLogger()
            self.text_logger.setLevel(logging.INFO)
            if mode == 'train':
                fh = logging.FileHandler(os.path.join(save_root, 'train.log'))
            else:
                fh = logging.FileHandler(os.path.join(save_root, 'test_{}_{}.log'.format(samplings_step, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))))
            self.text_logger.addHandler(fh)

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}.pth'.format(module_name, epoch)))

    def load_checkpoints(self, epoch, map_location="cpu"):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            path = osp.join(self.models_save_dir, '{}-{}.pth'.format(module_name, epoch))
            if module_name == 'optimizer':
                continue
            if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
                module.module.load_state_dict(load_network(path, map_location), strict=False)
            else:
                module.load_state_dict(load_network(path, map_location))
            if self.local_rank == 0:
                self.text_logger.info('loading weights')
                self.text_logger.info(path)

    def load_pretrain(self, pretrain_path, map_location="cpu"):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if 'ema' in module_name:
                path = pretrain_path.replace('model_', 'ema-model_')
            else:
                path = pretrain_path
            if module_name == 'optimizer':
                continue
            if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
                module.module.load_state_dict(load_network(path, map_location), strict=False)
            else:
                module.load_state_dict(load_network(path, map_location))
            if self.local_rank == 0:
                self.text_logger.info('loading weights')
                self.text_logger.info(path)

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            output_str  += 'samplings:{}'.format(self.samplimgs_step)
            self.text_logger.info(output_str)


    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.png'.format(n_iter, self.local_rank, sample_type)), nrow=1)
        
    def save_diff_image(self, image, diff, j, n_iter):
        cv2.imwrite(osp.join(self.images_save_dir, 'image_{}_{}.png'.format(n_iter, j)), image)
        # cv2.imwrite(osp.join(self.images_save_dir, 'diff_{}_{}.png'.format(n_iter, j)), diff)
        
        plt.imshow(diff, cmap='jet')
        plt.savefig(osp.join(self.images_save_dir, 'diff_{}_{}.png'.format(n_iter, j)))