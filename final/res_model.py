import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
  

class MinibatchSDLayer(nn.Module):
    def __init__(self):
        super(MinibatchSDLayer, self).__init__()
        
    def forward(self, x):
        mean_batch_std = x.std(0).mean()
        mean_batch_std = mean_batch_std.expand(x.size(0), 1, x.size(-1), x.size(-1))
        return torch.cat([x, mean_batch_std], 1)

class PixelWiseFeatureNormLayer(nn.Module):
    def __init__(self):
        super(PixelWiseFeatureNormLayer, self).__init__()
        
    def forward(self, x):
        return x / x.norm(2,1)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_()
        
    _u = u
    for _ in range(Ip):
        _v = l2normalize(torch.matmul(_u, W), eps=1e-12)
        _u = l2normalize(torch.matmul(_v, torch.transpose(W, 0, 1)), eps=1e-12)
        
    sigma = torch.sum(F.linear(_u, torch.transpose(W, 0, 1)) * _v)
    return sigma, _u
    
    
class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias)
        self.u = nn.Parameter(torch.Tensor(1, out_channels).normal_(), requires_grad=False)
        #self.u = torch.Tensor(1, out_channels).normal_()

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1).data
        sigma, _u = max_singular_value(w_mat, self.u.data)
        self.u.data = _u
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride, self.padding, self.dilation, self.groups)


def block(in_fm, out_fm):
    l = [
        SNConv2d(in_fm, out_fm, 3, 1, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        SNConv2d(out_fm, out_fm, 3, 1, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    return nn.Sequential(*l)
        

class ResidualGenerator(nn.Module):
    def __init__(self, zdim=100, n_feature_maps=256):
        super(ResidualGenerator, self).__init__()

        self.from_latent = nn.ConvTranspose2d(zdim, 2*n_feature_maps, 4, 1, 0, bias=False)
        self.block1 = block(2*n_feature_maps, 2*n_feature_maps)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.block2 = block(2*n_feature_maps, n_feature_maps)
        self.skip = nn.Conv2d(2*n_feature_maps,n_feature_maps, 1, 1, 0, bias=False)
        self.block3 = block(n_feature_maps, n_feature_maps)
        self.to_rgb = block(n_feature_maps, 3)
                
    def forward(self, x):
        x = self.from_latent(x) # 4x4x2nc
        x = self.upsample(x)
        x = self.block1(x) + x # 8x8x2nc
        x = self.upsample(x)
        x = self.block2(x) + self.skip(x) #16x16xnc
        x = self.upsample(x)
        x = self.block3(x) + x #32x32xnc
        x = self.upsample(x)
        x = self.to_rgb(x) #64x64x3
        return F.tanh(x)


class ResidualCritic(nn.Module):
    def __init__(self, zdim=100, n_feature_maps=256):
        super(ResidualCritic, self).__init__()

        self.from_rgb = block(3, n_feature_maps)
        self.downsample = nn.AvgPool2d(2)
        self.block1 = block(n_feature_maps, n_feature_maps)
        self.block2 = block(n_feature_maps, 2*n_feature_maps)
        self.skip2 = SNConv2d(n_feature_maps,2*n_feature_maps, 1, 1, 0, bias=False)
        self.block3 = block(2*n_feature_maps, 2*n_feature_maps)
        self.block4= block(2*n_feature_maps, 2*n_feature_maps)
        self.mbstd = MinibatchSDLayer()
        self.last_conv = SNConv2d(2*n_feature_maps+1, 4*n_feature_maps, 4, 1, 0, bias=False)
        self.fc = SNConv2d(4*n_feature_maps, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.from_rgb(x) #64x64xnc
        x = self.downsample(x)
        x = self.block1(x) + x #32x32xnc
        x = self.downsample(x)
        x = self.block2(x) + self.skip2(x) #16x16x2nc
        x = self.downsample(x)
        x = self.block3(x) + x #8x8x2nc
        x = self.downsample(x)
        x = self.block4(x) + x #4x4xnc
        x = self.mbstd(x)
        x = self.last_conv(x) #1x1x4nc
        x = self.fc(x) #1x1x1
        return x.view(-1,1).squeeze(1)