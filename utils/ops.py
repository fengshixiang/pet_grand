from torch import nn
import torch
import os.path as osp
import os
from torchvision.utils import save_image
import torch.distributed as dist
import inspect

def turn_on_spectral_norm(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        if module.out_channels != 1 and module.in_channels > 4:
            module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    else:
        world_size = 1
    if world_size is not None:
        rt /= world_size
    return rt


def load_network(state_dict, map_location='cpu'):
    if isinstance(state_dict, str):
        # if 'optimizer' in state_dict:
        #      optimizer_load = True
        # else:
        #      optimizer_load = False
        state_dict = torch.load(state_dict, map_location)
    # try: # debug
    #     del state_dict['betas']
    #     del state_dict['alphas_cumprod']
    #     del state_dict['alphas_cumprod_prev']
    #     del state_dict['sqrt_alphas_cumprod']
    #     del state_dict['sqrt_one_minus_alphas_cumprod']
    #     del state_dict['log_one_minus_alphas_cumprod']
    #     del state_dict['sqrt_recip_alphas_cumprod']
    #     del state_dict['sqrt_recipm1_alphas_cumprod']
    #     del state_dict['posterior_variance']
    #     del state_dict['posterior_log_variance_clipped']
    #     del state_dict['posterior_mean_coef1']
    #     del state_dict['posterior_mean_coef2']
    #     del state_dict['loss_weight']
    # except:
    #     pass
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        # if optimizer_load:
        #     new_state_dict[namekey] = v
        # if 'denoise_fn' in namekey:
        #     new_state_dict[namekey] = v
        new_state_dict[namekey] = v
    return new_state_dict