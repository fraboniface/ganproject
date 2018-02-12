import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tqdm import tqdm

SAVE_FOLDER = '../results/samples/cifar/'

gpu = torch.cuda.is_available()

transform = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
cifar = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform)

batch_size = 100
dataloader = torch.utils.data.DataLoader(cifar, batch_size=batch_size, shuffle=True, num_workers=2)

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
            # 32x32
            nn.Conv2d(n_feature_maps, 3, 1, 1, 0, bias=False), #1x1 convolution to reduce the number of feature maps and keep the same size
            nn.Tanh()
        )
        
    def  forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self, n_feature_maps=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #32x32
            nn.Conv2d(3, n_feature_maps, 4, 2, 1, bias=False),
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
            nn.Conv2d(4*n_feature_maps, 1, 4, 1, 0, bias=False),
            # 1x1
            nn.Sigmoid()
        )
        
    def  forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


z_size = 100
G = Generator(z_size)
G.apply(weights_init)

D = Discriminator()
D.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
g_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
d_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

criterion = nn.BCELoss()

#ones = Variable(torch.ones(batch_size))
#use line below for one-sided label smoothing
ones = 0.9*Variable(torch.ones(batch_size))
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


n_epochs = 50
for epoch in tqdm(range(1,n_epochs+1)):
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

	fake = G(fixed_noise)
	vutils.save_image(fake.data, '{}samples_epoch_{}.png'.format(SAVE_FOLDER, epoch), normalize=True, nrow=10)