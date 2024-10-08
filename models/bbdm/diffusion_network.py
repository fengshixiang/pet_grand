import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import math
import copy
from einops import rearrange

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #Weight initialization
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                                    

    def forward(self, x):
        return self.conv2(self.activate(self.conv1(x))) + x
    
class Block_spect(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(Block_spect, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                                    

    def forward(self, x):
        return self.conv2(self.activate(self.conv1(x)))

class UNetRes(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, nc=[64, 128, 256, 512], nb=4, bias=True):
        super(UNetRes, self).__init__()

        self.m_head = nn.Conv2d(in_channels=in_channels, out_channels=nc[0], kernel_size=3, stride=1, padding=1, bias=True)
        dim=32

        self.m_down1 = nn.Sequential(*[Block(nc[0], nc[0], bias=bias) for _ in range(nb)],
                                     nn.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, padding=0, bias=True))
        self.mlp_down1 = nn.Sequential(nn.GELU(),nn.Linear(dim, nc[1]))
        self.m_down2 = nn.Sequential(*[Block(nc[1], nc[1], bias=bias) for _ in range(nb)],
                                     nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, padding=0, bias=True))
        self.mlp_down2 = nn.Sequential(nn.GELU(),nn.Linear(dim, nc[2]))
        self.m_down3 = nn.Sequential(*[Block(nc[2], nc[2], bias=bias) for _ in range(nb)],
                                     nn.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, padding=0, bias=True))
        self.mlp_down3 = nn.Sequential(nn.GELU(),nn.Linear(dim, nc[3]))
        self.m_body = nn.Sequential(*[Block(nc[3], nc[3], bias=bias) for _ in range(nb)])
        self.mlp_body = nn.Sequential(nn.GELU(),nn.Linear(dim, nc[3]))


        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2, padding=0, bias=True),
                                   *[Block(nc[2], nc[2], bias=bias) for _ in range(nb)])
        self.mlp_up3 = nn.Sequential(nn.GELU(),nn.Linear(dim, nc[2]))
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2, padding=0, bias=True),
                                   *[Block(nc[1], nc[1], bias=bias) for _ in range(nb)])
        self.mlp_up2 = nn.Sequential(nn.GELU(),nn.Linear(dim, nc[1]))
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2, padding=0, bias=True),
                                   *[Block(nc[0], nc[0], bias=bias) for _ in range(nb)])

        self.m_tail = nn.Conv2d(nc[0], out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        

    def time_condition(self, mlp, feature, time_emb):
        condition = mlp(time_emb)
        condition = rearrange(condition, 'b c -> b c 1 1')
        return feature + condition

    def forward(self, x0, t, img_cond):
        x0 = torch.cat([x0, img_cond], 1)
        time_emb = self.time_mlp(t)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x2 = self.time_condition(self.mlp_down1, x2, time_emb) # t condition
        x3 = self.m_down2(x2)
        x3 = self.time_condition(self.mlp_down2, x3, time_emb) # t condition
        x4 = self.m_down3(x3)
        x4 = self.time_condition(self.mlp_down3, x4, time_emb) # t condition
        x = self.m_body(x4)
        x = self.time_condition(self.mlp_body, x, time_emb) # t condition
        x = self.m_up3(x+x4)
        x = self.time_condition(self.mlp_up3, x, time_emb) # t condition
        del x4
        x = self.m_up2(x+x3)
        x = self.time_condition(self.mlp_up2, x, time_emb) # t condition
        del x3
        x = self.m_up1(x+x2)
        del x2
        x = self.m_tail(x+x1) ## remove x1
        del x1
        return x

    
class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, context=True):
        super(Network, self).__init__()
        self.unet = UNetRes(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, timesteps, context):
        out = self.unet(x, timesteps, context)
        return out
