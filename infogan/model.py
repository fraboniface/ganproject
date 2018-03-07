
import numpy as np
import torch
import torch.nn as nn

class Generator32(nn.Module):
    def __init__(self, zdim, n_feature_maps, n_channels):
        super(Generator32, self).__init__()
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
        
    def  forward(self, x, mode='D'):
        
        x = self.main(x)
        source = self.D(x).view(-1, 1).squeeze(1)

        if mode == 'Q':
            Q_c_given_x = self.Q(x).view(-1, self.code.param_size)
            return x, source, Q_c_given_x
        else:
            # training on real images
            return x, source


class Generator64(nn.Module):
    def __init__(self, zdim, n_feature_maps, n_channels):
        super(Generator64, self).__init__()
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
            Q_c_given_x = self.Q(x).view(-1, self.code.param_size)
            return x, source, Q_c_given_x
        else:
            # training on real images
            return x, source



class Code:
    def __init__(self, discrete_list, n_continuous=0, type_continuous='uniform'):
        """
        Class representing the code of InfoGAN, i.e. the meaningful part of the latent representation.
        There are 2 parts to consider: the discrete random variables and the continuous ones.
        
        discrete_list: python list containing number of possibilities for each discrete distribution, e.g [10, 2, 5]
        n_continuous: number of continuous dimensions in the code
        type_continuous: whether the continuous variables follow a uniform or normal distribution
        """
        self.discrete_list = discrete_list
        self.dis_dim = len(discrete_list)
        self.dis_latent_size = 0
        self.dis_entropy = 0
        for k in self.discrete_list:
            self.dis_latent_size += k # onehot encoding
            self.dis_entropy += np.log(k) # the entropy of a discrete uniform distribution with k possibilities is log(k)
            
        self.dis_param_size = self.dis_latent_size
        
        self.type_continuous = type_continuous
        self.con_dim = n_continuous
        self.con_latent_size = n_continuous
        self.con_param_size = 2*n_continuous # mean and variance for each variable
        if type_continuous == 'normal':
            self.con_entropy = n_continuous/2 * np.log(2*np.pi*np.e) # entropy of N(0,1) x number of variables
        else:
            self.con_entropy = n_continuous * np.log(2) # entropy of  U([-1,1]) x number of variables
        
        self.dimension = self.dis_dim + self.con_dim # dimension in maths
        self.latent_size = self.dis_latent_size + self.con_latent_size # dimension in latent space (with onehot encoding)
        self.param_size = self.dis_param_size + self.con_param_size # dimension needed to specify distribution
        self.entropy = self.dis_entropy + self.con_entropy
            
    def sample_discrete(self, batch_size):
        c = None
        for k in self.discrete_list:
            tmp = np.random.multinomial(1, k*[1/k], size=batch_size)
            tmp = torch.from_numpy(tmp.astype(np.float32))
            if c is None:
                c = tmp
            else:
                c = torch.cat([c,tmp],1)
                
        return c
    
    def sample_continuous(self, batch_size):
        if self.con_dim > 0:
            if self.type_continuous == 'uniform':
                c = torch.FloatTensor(batch_size, self.con_dim).uniform_(-1,1)
            else:
                c = torch.FloatTensor(batch_size, self.con_dim).normal_(0,1)
            
            return c
    
    def sample(self, batch_size):
        dis = self.sample_discrete(batch_size)
        con = self.sample_continuous(batch_size)
        if self.dis_dim == 0:
            return con
        elif self.con_dim == 0:
            return dis
        else:
            return torch.cat([dis, con], 1)

    def get_logits(self, params):
        logits_list = []
        start_idx = 0
        for k in self.discrete_list:
            logits_list.append(params[:,start_idx:start_idx+k])
            start_idx += k

        return logits_list

    def get_gaussian_params(self, params):
        mean_end_idx = self.dis_param_size + self.con_dim
        mean = params[:,self.dis_param_size:mean_end_idx]
        log_var = params[:,mean_end_idx:]
        return mean, torch.exp(log_var)
