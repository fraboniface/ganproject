import torch
import torch.nn as nn

class Generator32(nn.Module):
    def __init__(self, latent_dim, n_feature_maps, n_channels):
        super(Generator32, self).__init__()
        self.main = nn.Sequential(
        	#1x1
        	nn.ConvTranspose2d(latent_dim, 8*n_feature_maps, 4, 1, 0, bias=False),
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
            # 32x32
            nn.Conv2d(n_feature_maps, n_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )
        
    def  forward(self, z, c):
        x = torch.cat([z,c],1)
        return self.main(x)

    
class D_and_Q_32(nn.Module):
    def __init__(self, n_feature_maps, n_channels, code):
        super(D_and_Q_32, self).__init__()
        self.code = code
        self.main = nn.Sequential(
            #32x32
            nn.Conv2d(n_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            nn.Conv2d(n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #8x8
            nn.Conv2d(2*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            nn.Conv2d(4*n_feature_maps, n_feature_maps, 4, 1, 0, bias=False),
            # 1x1
        )
        self.D = nn.Sequential(
	        	nn.Conv2d(n_feature_maps, 1, 1, 1, 0, bias=False),
	        	nn.Sigmoid()
	        )
        self.Q = nn.Conv2d(n_feature_maps, self.code.param_size, 1, 1, 0, bias=False)
        
    def  forward(self, x, mode='Q'):
        
        x = self.main(x)
        source = self.D(x).view(-1, 1).squeeze(1)

        if mode == 'Q':
            Qc_x = self.Q(x).view(-1, self.code.param_size)
            return source, Qc_x, x
        else:
            # training on real images
            return source, x


class Generator64(nn.Module):
    def __init__(self, latent_dim, n_feature_maps, n_channels):
        super(Generator64, self).__init__()
        self.main = nn.Sequential(
            #1x1
            nn.ConvTranspose2d(latent_dim, 8*n_feature_maps, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(n_feature_maps, n_channels, 4, 2, 1, bias=False),
            #64x64
            nn.Tanh()
        )
        
    def  forward(self, z, c):
        x = torch.cat([z,c],1)
        return self.main(x)

    
class D_and_Q_64(nn.Module):
    def __init__(self, n_feature_maps, n_channels, code):
        super(D_and_Q_64, self).__init__()
        self.code = code
        self.main = nn.Sequential(
            #64x64
            nn.Conv2d(n_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #32x32
            nn.Conv2d(n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            nn.Conv2d(2*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #8x8
            nn.Conv2d(4*n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            nn.Conv2d(2*n_feature_maps, n_feature_maps, 4, 1, 0, bias=False),
            # 1x1
        )
        self.D = nn.Sequential(
                nn.Conv2d(n_feature_maps, 1, 1, 1, 0, bias=False),
                nn.Sigmoid()
            )
        self.Q = nn.Conv2d(n_feature_maps, self.code.param_size, 1, 1, 0, bias=False)
        
    def  forward(self, x, mode='D'):
        
        x = self.main(x) # we return this for feature matching
        source = self.D(x).view(-1, 1).squeeze(1)

        if mode == 'Q':
            Qc_x = self.Q(x).view(-1, self.code.param_size)
            return source, Qc_x, x
        else:
            # training on real images
            return source, x