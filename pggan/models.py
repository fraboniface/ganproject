import torch
import torch.nn as nn
import torch.nn.functional as F

class GrowingGenerator(nn.Module):
    def __init__(self, zdim=100, init_size=4, final_size=128, n_feature_maps=128):
        super(GrowingGenerator, self).__init__()
       
        self.init_size = init_size
        self.final_size = final_size
        init_nfm = 8*n_feature_maps
        
        self.layers = [
            #1x1
            nn.ConvTranspose2d(zdim, init_nfm, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            nn.Conv2d(init_nfm, init_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            #4x4
        ]
        self.main = nn.Sequential(*self.layers)
        
        self.to_rgb = nn.Conv2d(init_nfm, 3, 1, 1, 0, bias=False)
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
            nn.Conv2d(self.current_nfm, future_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(future_nfm, future_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.layers += block
        self.main = nn.Sequential(*self.layers)
        
        self.current_size *= 2
        self.current_nfm = future_nfm
        self.to_rgb = nn.Conv2d(self.current_nfm, 3, 1, 1, 0, bias=False)
        
        self.new_parameters = nn.Sequential(*block).parameters()
        
        
class GrowingDiscriminator(nn.Module):
    def __init__(self, init_size=4, final_size=128, n_feature_maps=128):
        super(GrowingDiscriminator, self).__init__()
        self.init_size = init_size
        self.final_size = final_size
        init_nfm = 8 * n_feature_maps
        
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, init_nfm, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layers = [
            #4x4
            nn.Conv2d(init_nfm, init_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            #nn.Conv2d(init_nfm, init_nfm, 4, 1, 0, bias=False),
            nn.Conv2d(init_nfm, 1, 4, 1, 0, bias=False),
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
            nn.Conv2d(future_nfm, future_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(future_nfm, self.current_nfm, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        ]
        self.layers = block + self.layers
        self.main = nn.Sequential(*self.layers)
        
        self.current_size *= 2
        self.current_nfm = future_nfm
        self.from_rgb = nn.Conv2d(3, self.current_nfm, 1, 1, 0, bias=False)
        
        self.new_parameters = nn.Sequential(*block).parameters()