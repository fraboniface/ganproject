import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tqdm import tqdm

#SAVE_FOLDER = '/home/s1782330/samples_DCGAN_MNIST
SAVE_FOLDER = './saved_samples'
gpu = torch.cuda.is_available()

transform = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
cifar = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform)

batch_size = 50
dataloader = torch.utils.data.DataLoader(cifar, batch_size=batch_size, shuffle=True, num_workers=2)

class Generator(nn.Module):
    def __init__(self, zdim=100, n_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        	#1x1
        	nn.ConvTranspose2d(zdim, 8*n_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8*n_channels),
            nn.ReLU(True),
            #4x4
            nn.ConvTranspose2d(8*n_channels, 4*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*n_channels),
            nn.ReLU(True),
            #8x8
            nn.ConvTranspose2d(4*n_channels, 2*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*n_channels),
            nn.ReLU(True),
            #16x16
            nn.ConvTranspose2d(2*n_channels, n_channels, 4, 2, 1, bias=False),
            #32x32
            nn.Tanh()
        )
        
    def  forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #32x32
            nn.Conv2d(n_channels, 2*n_channels, 4, 2, 1),
            nn.BatchNorm2d(2*n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            nn.Conv2d(2*n_channels, 4*n_channels, 4, 2, 1),
            nn.BatchNorm2d(4*n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            #8x8
            nn.Conv2d(4*n_channels, 8*n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            #4x4
            nn.Conv2d(8*n_channels, 1, 4, 1, 0, bias=False),
            # 1x1
            nn.Sigmoid()
        )
        
    def  forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


n_epochs = 100
lr = 1e-4
z_size = 100
n_channels = 3
G = Generator(z_size, n_channels)
D = Discriminator(n_channels)

g_optimiser = optim.Adam(G.parameters(), lr=lr)
d_optimiser = optim.Adam(D.parameters(), lr=lr)

criterion = nn.BCELoss()

# modify for label smoothing
ones = Variable(torch.ones(batch_size))
zeros = Variable(torch.zeros(batch_size))

fixed_noise = Variable(torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1))

# to GPU
if gpu:
	G.cuda()
	D.cuda()
	criterion.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_noise = fixed_noise.cuda()

for epoch in tqdm(range(1,n_epochs+1)):
	fake = G(fixed_noise)
	vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (SAVE_FOLDER, epoch), normalize=True)

	for i, data in enumerate(dataloader):
		img, _ = data
		if gpu:
			img = img.cuda()

		img = Variable(img)

		# discriminator step
		D.zero_grad()
		#real data
		d_real = D(img)
		d_real_error = criterion(d_real, ones)
		d_real_error.backward()
		d_x = d_real.data.mean()
		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		if gpu:
			z = z.cuda()

		z = Variable(z)
		fake_data = G(z)
		d_fake = D(fake_data.detach())
		d_fake_error = criterion(d_fake, zeros)
		d_fake_error.backward()
		loss_d = d_real_error + d_fake_error
		d_g_z1 = d_fake.data.mean()
		d_optimiser.step()

		# generator step
		G.zero_grad()
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		if gpu:
			z = z.cuda()

		z = Variable(z)
		gen_data = G(z)
		d_output = D(gen_data)
		d_g_z2 = d_output.data.mean()
		g_error = criterion(d_output, ones)
		g_error.backward()
		g_optimiser.step()

		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
		% (epoch, n_epochs, i, len(dataloader),
		loss_d.data[0], g_error.data[0], d_x, d_g_z1, d_g_z2))