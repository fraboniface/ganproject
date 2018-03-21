
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision  import transforms, datasets
import torchvision.utils as vutils

import pickle
import sys
from tqdm import tqdm
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
  

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


model_name = 'SNDCGAN_fm_BN2'
dataset_name = 'portraits64'

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/regular_save/'

batch_size = 100

img_size = 64
n_channels = 3
n_feature_maps = 128
n_epochs = 70

transform = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


dataset = datasets.ImageFolder('../portraits64/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#custom weights init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
            nn.ConvTranspose2d(n_feature_maps, n_channels, 4, 2, 1, bias=False),
            #64x64
            nn.Tanh()
        )
        
    def  forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, n_feature_maps=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #64x64
            SNConv2d(n_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #32x32
            SNConv2d(n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            SNConv2d(2*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            #8x8
            SNConv2d(4*n_feature_maps, 8*n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*n_feature_maps),
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


z_size = 100
G = Generator(z_size, n_feature_maps)
G.apply(weights_init)

D = Discriminator(n_feature_maps)
D.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=lr, betas=(beta1, beta2))

fixed_z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
fixed_z = Variable(fixed_z, volatile=True)

#ones = Variable(torch.ones(batch_size))
ones = Variable(torch.FloatTensor(batch_size).uniform_(0.9,1)) # label smoothing
zeros = Variable(torch.zeros(batch_size))

D_criterion = nn.BCELoss()
G_criterion = nn.MSELoss()

# to GPU
gpu = torch.cuda.is_available()
if gpu:
    G.cuda()
    D.cuda()
    D_criterion.cuda()
    G_criterion.cuda()
    ones = ones.cuda()
    zeros = zeros.cuda()
    fixed_z = fixed_z.cuda()

results = {
    'D_real_loss': [],
    'D_fake_loss': [],
    'D_loss': [],
    'G_loss': []

}

for epoch in tqdm(range(1,n_epochs+1)):
    for img, labels in dataloader:
        if img.size(0) < batch_size:
            continue
        if gpu:
            img = img.cuda()
            #labels = labels.cuda()

        img = Variable(img)
        #labels = Variable(labels)

        # DISCRIMINATOR STEP
        D.zero_grad()

        #real data
        D_real = D(img)
        D_real_error = D_criterion(D_real, ones)
        results['D_real_loss'].append(D_real_error)
        D_real_error.backward()

        #fake data
        z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
        if gpu:
            z = z.cuda()

        z = Variable(z)
        fake_data = G(z)
        D_fake = D(fake_data.detach())
        D_fake_error = D_criterion(D_fake, zeros)
        results['D_fake_loss'].append(D_fake_error.data.cpu().numpy())
        D_fake_error.backward()

        results['D_loss'].append(D_real_error+D_fake_error)
        D_optimiser.step()


        # GENERATOR STEP
        G.zero_grad()

        z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
        if gpu:
            z = z.cuda()

        z = Variable(z)
        gen_data = G(z)

        real_features = D(img, matching=True)
        fake_features = D(gen_data, matching=True)
        real_mean = torch.mean(real_features, 0)
        fake_mean = torch.mean(fake_features, 0)
        G_error = G_criterion(fake_mean, real_mean.detach())

        results['G_loss'].append(G_error.data.cpu().numpy())
        G_error.backward()

        G_optimiser.step()

    fake = G(fixed_z)
    vutils.save_image(fake.data, '{}{}_{}_samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, n_feature_maps, epoch), normalize=True, nrow=10)

    if epoch %5 == 0:
    # generates samples with fixed noise
                # saves everything, overwriting previous epochs
        torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_generator'.format(dataset_name, model_name, epoch))
        torch.save(D.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_discriminator'.format(dataset_name, model_name, epoch))

        with open(RESULTS_FOLDER + 'losses_and_samples_{}_{}_epoch_{}.p'.format(dataset_name, model_name, epoch), 'wb') as f:
            pickle.dump(results, f)
