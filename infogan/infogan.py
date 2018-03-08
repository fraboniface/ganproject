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

from models import *
from utils import Code

model_name = 'InfoGAN'

dataset_name = sys.argv[1]
assert dataset_name in ['paintings64', 'mnist', 'cifar']

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'

batch_size = 100

if dataset_name == 'paintings64':
	n_classes = 5
	img_size = 64
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 50
	code = Code(0, 10, 'uniform')
	lambda_param = 1e-2

elif dataset_name == 'cifar':
	n_classes = 10
	img_size = 32
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 50
	code = Code(10, 10, 'uniform')
	lambda_param = 1e-2
	
elif dataset_name == 'mnist':
	n_classes = 10
	img_size = 32
	n_channels = 1
	n_feature_maps = 64
	n_epochs = 20
	code = Code(10, 5, 'uniform')
	lambda_param = 0.1


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
	# take into account class imbalance by using a weighted sampler
	class_counts = [4089,10983,11545,12926,5702]
	weights = 1 / torch.Tensor(class_counts)
	weights = weights.double()
	sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
	dataset = datasets.ImageFolder('../paintings64/', transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)

else:
	if dataset_name == 'cifar':
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

latent_size = 100
z_size = latent_size - code.dimension
effective_latent_size = z_size + code.latent_size

if dataset_name == 'paintings64':
	G = Generator64(effective_latent_size, n_feature_maps, n_channels)
	DQ = D_and_Q_64(n_feature_maps, n_channels, code)
else:
	G = Generator32(effective_latent_size, n_feature_maps, n_channels)
	DQ = D_and_Q_32(n_feature_maps, n_channels, code)

G.apply(weights_init)	
DQ.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
DQ_optimiser = optim.Adam(DQ.parameters(), lr=lr, betas=(beta1, beta2))

fixed_z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
fixed_c = code.sample(batch_size)
fixed_z = Variable(fixed_z, volatile=True)
fixed_c = Variable(fixed_c, volatile=True)

#ones = Variable(torch.ones(batch_size))
ones = Variable(torch.FloatTensor(batch_size).uniform_(0.9,1)) # label smoothing
zeros = Variable(torch.zeros(batch_size))

source_criterion = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

def log_gaussian(mean, var, x):
	"""Arguments are vectors"""
	log = -1/2 * torch.log(2*np.pi*var) - (x-mean)**2 / (2*var)
	return log.sum(1).mean()

def compute_milb(Qc_x, c, code):
	"""Computes the mutual information lower bound"""

	milb = 0
	#milb = code.entropy # for true lower bound
	if code.n_classes > :
		dis_c_onehot = code.get_logits(c)
		_, dis_c_num = dis_c_onehot.max(1) # CrossEntropyLoss wants numerical targets, not onehot
		Q_logits = code.get_logits(Qc_x)
		milb -= ce_loss(Q_logits, dis_c) # note the minus

	if code.n_continuous > 0:
		con_c = code.get_gaussian_values(c)
		mean, var = code.get_gaussian_params(Qc_x)
		milb += log_gaussian(mean, var, con_c)

	return milb

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	DQ.cuda()
	source_criterion.cuda()
	ce_loss.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_z = fixed_z.cuda()
	fixed_c = fixed_c.cuda()


for epoch in tqdm(range(1,n_epochs+1)):
	for img, labels in dataloader:
		if img.size(0) < batch_size:
			continue
		if gpu:
			img = img.cuda()
			#labels = labels.cuda()

		img = Variable(img)
		#labels = Variable(labels)

		# DISCRIMINATOR and Q STEP
		DQ.zero_grad()

		#real data
		D_real = DQ(img, mode='D')
		D_real_error = source_criterion(D_real, ones)

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		c = code.sample(batch_size)
		if gpu:
			z = z.cuda()
			c = c.cuda()

		z = Variable(z)
		c = Variable(c)
		fake_data = G(z, c)
		D_fake, Qc_x = DQ(fake_data.detach(), mode='Q')
		D_fake_error = source_criterion(D_fake, zeros)

		milb = compute_milb(Qc_x, c, code)
		
		DQ_loss = D_real_error + D_fake_error - lambda_param * milb
		DQ_loss.backward(retain_graph=True)

		DQ_optimiser.step()


		# GENERATOR STEP
		G.zero_grad()

		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		c = code.sample(batch_size)
		if gpu:
			z = z.cuda()
			c = c.cuda()

		z = Variable(z)
		c = Variable(c)

		gen_data = G(z, c)
		D_gen, Qc_x = DQ(gen_data, mode='Q')
		G_error = source_criterion(D_gen, ones)

		milb = compute_milb(Qc_x, c, code)

		G_loss = G_error - lambda_param * milb
		G_loss.backward()

		G_optimiser.step()

	# generates samples with fixed noise
	fake = G(fixed_z, fixed_c)
	vutils.save_image(fake.data, '{}{}_{}_samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, n_feature_maps, epoch), normalize=True, nrow=10)

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_{}_generator'.format(dataset_name, model_name, n_feature_maps))
	torch.save(DQ.state_dict(), RESULTS_FOLDER + '{}_{}_{}_D_and_Q'.format(dataset_name, model_name, n_feature_maps))