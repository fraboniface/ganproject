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

model_name = 'DCGAN_FM'

dataset_name = sys.argv[1]
assert dataset_name in ['paintings64', 'mnist', 'cifar']

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'

batch_size = 100

if dataset_name == 'paintings64':
	img_size = 64
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 70

elif dataset_name == 'cifar':
	img_size = 32
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 50
	
elif dataset_name == 'mnist':
	img_size = 32
	n_channels = 1
	n_feature_maps = 64
	n_epochs = 20


if dataset_name == 'mnist':
	transform = transforms.Compose(
	[
		transforms.Resize(img_size),
	    transforms.ToTensor(),
	    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
	])
else:
	transform = transforms.Compose(
	[
	    transforms.ToTensor(),
	    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
	])


if dataset_name == 'paintings64':
	dataset = datasets.ImageFolder('../paintings64/', transform=transform)
elif dataset_name == 'cifar':
	dataset = datasets.CIFAR10('../data', train=True, transform=transform)
elif dataset_name == 'mnist':
	dataset = datasets.MNIST('../data', train=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


#custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

if dataset_name == 'paintings64':
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
	            nn.Conv2d(4*n_feature_maps, 8*n_feature_maps, 4, 2, 1, bias=False),
	            nn.BatchNorm2d(8*n_feature_maps),
	            nn.LeakyReLU(0.2, inplace=True),
	            # 4x4
	            )

	        self.output_layer = nn.Sequential(
	            nn.Conv2d(8*n_feature_maps, 1, 4, 1, 0, bias=False),
	            #1x1
	            nn.Sigmoid()
	        )
	        
	    def  forward(self, x):
	        x = self.main(x)
	        output = self.output_layer(x)
	        return x, output.view(-1, 1).squeeze(1)

else:
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
	            nn.Conv2d(n_feature_maps, n_channels, 1, 1, 0, bias=False),
	            nn.Tanh()
	        )
	        
	    def  forward(self, x):
	        return self.main(x)
	    
	class Discriminator(nn.Module):
	    def __init__(self, n_feature_maps=64):
	        super(Discriminator, self).__init__()
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
	            )

	        self.output_layer = nn.Sequential(
	            nn.Conv2d(4*n_feature_maps, 1, 4, 1, 0, bias=False),
	            #1x1
	            nn.Sigmoid()
	        )
	        
	    def  forward(self, x):
	        x = self.main(x)
	        output = self.output_layer(x)
	        return x, output.view(-1, 1).squeeze(1)


z_size = 100
G = Generator(z_size, n_feature_maps)
G.apply(weights_init)

D = Discriminator(n_feature_maps)
D.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

fixed_z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
fixed_z = Variable(fixed_z, volatile=True)

ones = 0.9*Variable(torch.ones(batch_size))
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
		layer_real, D_real = D(img)
		D_real_error = D_criterion(D_real, ones)
		results['D_real_loss'].append(D_real_error)
		D_real_error.backward()

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		if gpu:
			z = z.cuda()

		z = Variable(z)
		fake_data = G(z)
		_, D_fake = D(fake_data.detach())
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
		layer_fake, _ = D(gen_data)
		#G_error = criterion(D_gen, ones)
		G_error = G_criterion(layer_fake, layer_real.detach())
		results['G_loss'].append(G_error.data.cpu().numpy())
		G_error.backward()

		G_optimiser.step()


	# generates samples with fixed noise
	fake = G(fixed_z)
	vutils.save_image(fake.data, '{}{}{}_samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, n_feature_maps, epoch), normalize=True, nrow=10)

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_{}_generator'.format(dataset_name, model_name, n_feature_maps))
	torch.save(D.state_dict(), RESULTS_FOLDER + '{}_{}_{}_discriminator'.format(dataset_name, model_name, n_feature_maps))

	with open(RESULTS_FOLDER + 'losses_and_samples_{}_{}_{}.p'.format(dataset_name, model_name, n_feature_maps), 'wb') as f:
		pickle.dump(results, f)