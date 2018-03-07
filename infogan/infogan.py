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

from model import *


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
	code = Code([], 10, 'uniform')
	lambda_param = 0.5

elif dataset_name == 'cifar':
	n_classes = 10
	img_size = 32
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 50
	code = Code([], 10, 'uniform')
	lambda_param = 0.5
	
elif dataset_name == 'mnist':
	n_classes = 10
	img_size = 32
	n_channels = 1
	n_feature_maps = 64
	n_epochs = 20
	code = Code([10], 5, 'uniform')
	lambda_param = 0.5


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

ones = 0.9*Variable(torch.ones(batch_size))
zeros = Variable(torch.zeros(batch_size))



D_criterion = nn.BCELoss()
G_criterion = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

def log_gaussian(mean, var, x):
	"""Arguments are vectors"""
	print(var)
	log = -1/2 * torch.log(2*np.pi*var) - (x-mean)**2 / (2*var)
	return log.sum(1).mean()

def Q_con_criterion(Q_params, c):

	if code.con_dim > 0:
		mean, var = code.get_gaussian_params(Q_params)
		return log_gaussian(mean, var, c)

	else:
		return 0

def Q_dis_criterion(Q_params, c):

	Q_dis_error = 0
	if code.dis_dim > 0:
		Q_logits_list = code.get_logits(Q_params)
		dis_c_list = code.get_logits(dis_c)
		for logits, onehot_targets in zip(Q_logits_list, dis_c_list):
			# apply cross-entropy loss to all different discrete random variables
			_, targets = onehot_targets.max(1) # NLL wants numerical targets, not onehot
			Q_dis_error += cross_entropy_loss(logits, targets)
		
	return Q_dis_error

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	DQ.cuda()
	D_criterion.cuda()
	G_criterion.cuda()
	cross_entropy_loss.cuda()
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

		# DISCRIMINATOR STEP
		DQ.zero_grad()

		#real data
		layer_real, D_real = DQ(img)
		D_real_error = D_criterion(D_real, ones)
		D_real_error.backward()

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		c = code.sample(batch_size)
		if gpu:
			z = z.cuda()
			c = c.cuda()

		z = Variable(z)
		c = Variable(c)
		fake_data = G(z, c)
		_, D_fake = DQ(fake_data.detach())
		D_fake_error = D_criterion(D_fake, zeros)

		D_fake_error.backward()

		DQ_optimiser.step()


		# GENERATOR STEP
		G.zero_grad()
		DQ.zero_grad() # we're going to optimise it again

		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		dis_c = code.sample_discrete(batch_size)
		con_c = code.sample_continuous(batch_size)
		if gpu:
			z = z.cuda()
			if dis_c is not None:
				dis_c = dis_c.cuda()
			if con_c is not None:
				con_c = con_c.cuda()

		z = Variable(z)
		dis_c = Variable(dis_c)
		con_c = Variable(con_c)
		if dis_c is None:
			c = con_c
		elif con_c is None:
			c = dis_c
		else:
			c = torch.cat([dis_c, con_c], 1)

		gen_data = G(z, c)
		layer_fake, D_gen, Q_params = DQ(gen_data, mode='Q')
		G_error = G_criterion(layer_fake, layer_real.detach()) # feature matching
		Q_dis_error = Q_dis_criterion(Q_params, dis_c)
		Q_con_error = Q_con_criterion(Q_params, con_c)

		lower_bound = Q_dis_error + Q_con_error + code.entropy
		GQ_loss = - lambda_param * lower_bound

		GQ_loss.backward(retain_graph=True)
		G_error.backward()

		G_optimiser.step()
		DQ_optimiser.step()


	# generates samples with fixed noise
	fake = G(fixed_z, fixed_c)
	vutils.save_image(fake.data, '{}InfoGAN1_{}_samples_epoch_{}.png'.format(SAVE_FOLDER, n_feature_maps, epoch), normalize=True, nrow=10)

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}InfoGAN1{}_generator'.format(dataset_name, n_feature_maps))
	torch.save(DQ.state_dict(), RESULTS_FOLDER + '{}InfoGAN1{}_D_and_Q'.format(dataset_name, n_feature_maps))