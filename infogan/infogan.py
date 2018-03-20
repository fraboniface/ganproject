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

model_name = 'InfoGAN3'

dataset_name = sys.argv[1]
assert dataset_name in ['paintings64', 'mnist', 'cifar']

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/regular_save/'

batch_size = 100

if dataset_name == 'paintings64':
	n_classes = 5 # genres, but could be something else
	img_size = 64
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 60
	code = Code(0, 2, 'uniform')

elif dataset_name == 'cifar':
	n_classes = 10
	img_size = 32
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 50
	code = Code(10, 5, 'uniform')
	
elif dataset_name == 'mnist':
	n_classes = 10
	img_size = 32
	n_channels = 1
	n_feature_maps = 64
	n_epochs = 20
	code = Code(0, 3, 'uniform')

lambda_dis = 1
lambda_con = 0.1


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
	probs = []
	for i in class_counts:
	    probs += [i]*i
	weights = 1 / torch.Tensor(probs)
	weights = weights.double()
	dataset = datasets.ImageFolder('../paintings64/', transform=transform)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(dataset))
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
z_size = latent_size - code.latent_size

if dataset_name == 'paintings64':
	G = Generator64(latent_size, n_feature_maps, n_channels)
	DQ = D_and_Q_64(n_feature_maps, n_channels, code)
else:
	G = Generator32(latent_size, n_feature_maps, n_channels)
	DQ = D_and_Q_32(n_feature_maps, n_channels, code)

G.apply(weights_init)	
DQ.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
DQ_optimiser = optim.Adam(DQ.parameters(), lr=lr, betas=(beta1, beta2))

"""if dataset_name == 'mnist':
	fixed_z = torch.FloatTensor(10, latent_size - n_classes, 1, 1).normal_(0,1)
	fixed_z = fixed_z.repeat(5,1,1,1)
	onehot = torch.eye(n_classes).view(n_classes,n_classes)
	fixed_c = onehot.repeat(5,1)

else:
	fixed_z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
	fixed_c = code.sample(batch_size)
	fixed_z = Variable(fixed_z, volatile=True)
	fixed_c = Variable(fixed_c, volatile=True)"""

fixed_z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
fixed_c = code.sample(batch_size)
fixed_z = Variable(fixed_z, volatile=True)
fixed_c = Variable(fixed_c, volatile=True)


#ones = Variable(torch.ones(batch_size))
ones = Variable(torch.FloatTensor(batch_size).uniform_(0.9,1)) # label smoothing
zeros = Variable(torch.zeros(batch_size))

D_criterion = nn.BCELoss()
G_criterion = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

def log_gaussian(mean, var, x):
	"""Arguments are vectors"""
	log = -1/2 * torch.log(2*np.pi*var) - (x-mean)**2 / (2*var)
	return log.sum(1).mean()

def compute_milb(Qc_x, c, code, lambda_dis, lambda_con):
	"""Computes the mutual information lower bound"""

	milb_dis = 0
	milb_con = 0
	#milb = code.entropy # for true lower bound
	if code.n_classes > 0:
		dis_c_onehot = code.get_logits(c)
		_, dis_c_num = dis_c_onehot.max(1) # CrossEntropyLoss wants numerical targets, not onehot
		Q_logits = code.get_logits(Qc_x)
		milb_dis = - ce_loss(Q_logits, dis_c_num) # note the minus

	if code.n_continuous > 0:
		con_c = code.get_gaussian_values(c)
		mean, var = code.get_gaussian_params(Qc_x)
		milb_con = log_gaussian(mean, var, con_c)

	milb = milb_dis + milb_con + code.entropy
	Q_obj = lambda_dis*milb_dis + lambda_con*milb_con

	return milb, Q_obj

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	DQ.cuda()
	D_criterion.cuda()
	G_criterion.cuda()
	ce_loss.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_z = fixed_z.cuda()
	fixed_c = fixed_c.cuda()

train_hist = {}
train_hist['D_loss'] = []
train_hist['G_loss'] = []
train_hist['milb'] = []


for epoch in tqdm(range(1,n_epochs+1)):

	D_losses = []
	G_losses = []
	milbs = []

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
		D_real, features_real = DQ(img, mode='D')
		D_real_error = D_criterion(D_real, ones)

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		c = code.sample(batch_size)
		if gpu:
			z = z.cuda()
			c = c.cuda()

		z = Variable(z)
		c = Variable(c)
		fake_data = G(z, c)
		D_fake, Qc_x, _ = DQ(fake_data.detach(), mode='Q')
		D_fake_error = D_criterion(D_fake, zeros)

		milb, Q_obj = compute_milb(Qc_x, c, code, lambda_dis, lambda_con)
		
		D_error = D_real_error + D_fake_error
		DQ_loss = D_error - Q_obj

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
		D_gen, Qc_x, features_fake = DQ(gen_data, mode='Q')

		mean_real = torch.mean(features_real,0)
		mean_fake = torch.mean(features_fake, 0)

		G_error = G_criterion(mean_fake, mean_real.detach())

		milb, Q_obj = compute_milb(Qc_x, c, code, lambda_dis, lambda_con)

		G_loss = G_error - Q_obj

		G_loss.backward()
		G_optimiser.step()

		D_losses.append(D_error.data[0])
		G_losses.append(G_error.data[0])
		milbs.append(milb.data[0])

	if epoch % 5 == 0:

		# generates samples with fixed noise
		fake = G(fixed_z, fixed_c)
		vutils.save_image(fake.data, '{}{}_samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, epoch), normalize=True, nrow=10)

		train_hist['D_loss'].append(torch.mean(torch.FloatTensor(D_losses)))
		train_hist['G_loss'].append(torch.mean(torch.FloatTensor(G_losses)))
		train_hist['milb'].append(torch.mean(torch.FloatTensor(milbs)))

		with open(RESULTS_FOLDER + 'losses_{}_{}_epoch_{}.p'.format(dataset_name, model_name, epoch), 'wb') as f:
			pickle.dump(train_hist, f)

		# saves everything, overwriting previous epochs
		torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_generator'.format(dataset_name, model_name, epoch))
		torch.save(DQ.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_D_and_Q'.format(dataset_name, model_name, epoch))