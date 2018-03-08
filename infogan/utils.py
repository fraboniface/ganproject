import numpy as np
import torch

class Code:
    def __init__(self, n_classes, n_continuous, type_continuous='uniform'):
        """
        Class representing the code of InfoGAN, i.e. the meaningful part of the latent representation.
        There are 2 parts to consider: the discrete random variable and the continuous ones.
        
        n_classes: number of classes for the discrete random variables
        n_continuous: number of continuous dimensions in the code
        type_continuous: whether the continuous variables follow a uniform or normal distribution
        """

        self.n_classes = n_classes
        if n_classes > 0:
            self.dis_entropy = np.log(n_classes)
        else:
            self.dis_entropy = 0
        
        self.type_continuous = type_continuous
        self.n_continuous = n_continuous
        if type_continuous == 'normal':
            self.con_entropy = n_continuous/2 * np.log(2*np.pi*np.e) # entropy of N(0,1) x number of variables
        else:
            self.con_entropy = n_continuous * np.log(2) # entropy of  U([-1,1]) x number of variables
        
        self.latent_size = self.n_classes + self.n_continuous # dimension in latent space (with onehot encoding)
        self.param_size = self.n_classes + 2 * self.n_continuous # dimension needed to specify distribution (2 params per Gaussian)
        self.entropy = self.dis_entropy + self.con_entropy # distribution factorises so entropy is sum of entropies
            
    def sample_discrete(self, batch_size):
        if self.n_classes > 0:
            c = np.random.multinomial(1, self.n_classes*[1/self.n_classes], size=batch_size)
            return torch.from_numpy(c.astype(np.float32))
        else:
            return torch.Tensor([])
    
    def sample_continuous(self, batch_size):
        if self.n_continuous > 0:
            if self.type_continuous == 'uniform':
                c = torch.FloatTensor(batch_size, self.n_continuous).uniform_(-1,1)
            else:
                c = torch.FloatTensor(batch_size, self.n_continuous).normal_(0,1)
            
            return c
            
        else:
            return torch.Tensor([])
    
    def sample(self, batch_size):
        dis = self.sample_discrete(batch_size)
        con = self.sample_continuous(batch_size)
        return torch.cat([dis, con], 1)

    def get_logits(self, Qc_x):
        return Qc_x[:,:self.n_classes]

    def get_gaussian_values(self, c):
        return c[:,self.n_classes:]
        
    def get_gaussian_params(self, Qc_x):
        mean_end_idx = self.n_classes + self.n_continuous
        mean  = Qc_x[:, self.n_classes:mean_end_idx]
        log_var = Qc_x[:,mean_end_idx:]
        return mean, torch.exp(log_var)