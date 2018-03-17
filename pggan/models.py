import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair


def l2normalize(v, eps=1e-12):
    return v / (((v**2).sum())**0.5 + eps)

def max_singular_value(W, u=None, n_iter=1):
    """
    power iteration for weight parameter
    """
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1)
        if torch.cuda.is_available():
            u = u.cuda()
        
    for i in range(n_iter):
        v = l2normalize(torch.matmul(u, W))
        u = l2normalize(torch.matmul(v, torch.transpose(W, 0, 1)))
        
    sigma = torch.sum(F.linear(u, torch.transpose(W, 0, 1)) * v)
    return sigma, u
    

class MinibatchSDLayer(nn.Module):
    def __init__(self):
        super(MinibatchSDLayer, self).__init__()
        
    def forward(self, x):
        mean_batch_std = x.std(0).mean()
        mean_batch_std = mean_batch_std.expand(x.size(0), 1, x.size(-1), x.size(-1))
        return torch.cat([x, mean_batch_std], 1)
    
    
class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias)
        self.u = None

    @property
    def W_(self):
        w_mat = self.weight.data.view(self.weight.size(0), -1)
        sigma, u = max_singular_value(w_mat, self.u)
        self.u = u
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride, self.padding, self.dilation, self.groups)
        

class GrowingGenerator(nn.Module):
    def __init__(self, zdim=100, init_size=4, final_size=128, n_feature_maps=128):
        super(GrowingGenerator, self).__init__()
       
        self.init_size = init_size
        self.final_size = final_size
        init_nfm = 8*n_feature_maps
        
        self.layers = [
            #1x1
            nn.ConvTranspose2d(zdim, init_nfm, 4, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            nn.Conv2d(init_nfm, init_nfm, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
            #4x4
        ]
        self.main = nn.Sequential(*self.layers)
        
        self.to_rgb = nn.Conv2d(init_nfm, 3, 1, 1, 0, bias=True)
        self.current_size = init_size
        self.current_nfm = init_nfm
                
    def forward(self, x):
        x = self.main(x)
        x = self.to_rgb(x)
        return F.tanh(x)

    def grow(self):
        if self.current_size == self.final_size:
            print("Network can't grow more")
            return
        
        if self.current_size in [8,32]: # don't decrease everytime because otherwise it's too fast
            future_nfm = self.current_nfm
        else:
            future_nfm = int(self.current_nfm / 2)
            
        block = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.current_nfm, future_nfm, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(future_nfm, future_nfm, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.layers += block
        self.main = nn.Sequential(*self.layers)
        
        self.current_size *= 2
        self.current_nfm = future_nfm
        self.to_rgb = nn.Conv2d(self.current_nfm, 3, 1, 1, 0, bias=True)
        
        self.new_parameters = nn.Sequential(*block).parameters()
        
        
class SNGrowingDiscriminator(nn.Module):
    def __init__(self, init_size=4, final_size=128, n_feature_maps=128):
        super(SNGrowingDiscriminator, self).__init__()
        self.init_size = init_size
        self.final_size = final_size
        init_nfm = 8 * n_feature_maps
        
        self.from_rgb = nn.Sequential(
            SNConv2d(3, init_nfm, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layers = [
            MinibatchSDLayer(),
            #4x4
            SNConv2d(init_nfm+1, init_nfm, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            #nn.Conv2d(init_nfm, init_nfm, 4, 1, 0, bias=False),
            SNConv2d(init_nfm, 1, 4, 1, 0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            #1x1
            #nn.Conv2d(init_nfm, 1, 1, 1, 0, bias=False) # equivalent to fully connected
            #nn.Sigmoid()
        ]
        self.main = nn.Sequential(*self.layers)
        
        self.current_size = init_size
        self.current_nfm = init_nfm
        
    def forward(self, x):
        if x.size(3) != self.current_size:
            print("input is of the wrong size (should be {})".format(self.current_size))
            return
        
        x = self.from_rgb(x)
        output = self.main(x)
        return output.view(-1,1).squeeze()
    
    def grow(self):
        if self.current_size == self.final_size:
            print("Network can't grow more")
            return  
        
        if self.current_size in [8,32]:
            future_nfm = self.current_nfm
        else:
            future_nfm = int(self.current_nfm / 2)
        
        block = [
            SNConv2d(future_nfm, future_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(future_nfm, self.current_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        ]
        self.layers = block + self.layers
        self.main = nn.Sequential(*self.layers)
        
        self.current_size *= 2
        self.current_nfm = future_nfm
        self.from_rgb = SNConv2d(3, self.current_nfm, 1, 1, 0, bias=False)
        
        self.new_parameters = nn.Sequential(*block).parameters()


class Generator(nn.Module):
    def __init__(self, zdim=100, n_feature_maps=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #1x1
            nn.ConvTranspose2d(zdim, 8*n_feature_maps, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8*n_feature_maps),
            nn.ReLU(True),
            #4x4
            nn.ConvTranspose2d(8*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_feature_maps),
            nn.ReLU(True),
            #8x8
            nn.ConvTranspose2d(4*n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_feature_maps),
            nn.ReLU(True),
            #16x16
            nn.ConvTranspose2d(2*n_feature_maps, n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps),
            nn.ReLU(True),
            #32x32
            nn.ConvTranspose2d(n_feature_maps, 3, 4, 2, 1, bias=False),
            #64x64
            nn.Tanh()
        )
        
    def  forward(self, x):
        return self.main(x)


class SNDiscriminator(nn.Module):
    def __init__(self, n_feature_maps=64):
        super(SNDiscriminator, self).__init__()
        self.main = nn.Sequential(
            #64x64
            SNConv2d(3, n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #32x32
            SNConv2d(n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            SNConv2d(2*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #8x8
            SNConv2d(4*n_feature_maps, 8*n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.output = nn.Sequential(
            # 4x4
            SNConv2d(8*n_feature_maps, 1, 4, 1, 0, bias=False),
            #1x1
            nn.Sigmoid()
        )
        
    def  forward(self, x, matching=False):
        x = self.main(x)
        if matching :
            return x
        else:
            output = self.output(x)
            return output.view(-1, 1).squeeze(1)