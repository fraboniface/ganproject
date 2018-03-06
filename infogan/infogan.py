import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision  import transforms, datasets
import torchvision.utils as vutils

import pickle
from tqdm import tqdm

from model import *

dataset_name = 'mnist'

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'

batch_size = 100

img_size = 32
n_channels = 1
n_feature_maps = 64
n_epochs = 20
n_classes = 10
code = Code([n_classes])
lambda_param = 1

transform = transforms.Compose(
[
	transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

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
G = Generator(effective_latent_size, n_feature_maps, n_channels)
G.apply(weights_init)

DQ = D_and_Q(n_feature_maps, n_channels, code)
DQ.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
DQ_optimiser = optim.Adam(DQ.parameters(), lr=lr, betas=(beta1, beta2))

fixed_z = torch.FloatTensor(5, z_size, 1, 1).normal_(0,1)
fixed_z = fixed_z.repeat(n_classes,1,1,1)

onehot = torch.eye(n_classes).view(n_classes,n_classes,1,1)
fixed_c = onehot.repeat(5,1,1,1)

fixed_z = Variable(fixed_z, volatile=True)
fixed_c = Variable(fixed_c, volatile=True)

ones = 0.9*Variable(torch.ones(batch_size))
zeros = Variable(torch.zeros(batch_size))


def log_gaussian(x, mean, var):
	"""Arguments are vectors"""
	log = -(torch.log(2*np.pi*var) + (x-mean)**2/var)/2
	return log.sum(1).mean()

source_criterion = nn.BCELoss()
dis_criterion = nn.CrossEntropyLoss() # discrete
con_criterion = log_gaussian # continuous

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	DQ.cuda()
	source_criterion.cuda()
	dis_criterion.cuda()
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
		D_real_source = DQ(img)
		D_real_error = source_criterion(D_real_source, ones)
		D_real_error.backward()

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		dis_c = code.sample_discrete(batch_size)
		if gpu:
			z = z.cuda()
			dis_c = dis_c.cuda()

		z = Variable(z)
		dis_c = Variable(dis_c)
		fake_data = G(z, dis_c)
		D_fake = DQ(fake_data.detach())
		D_fake_error = source_criterion(D_fake, zeros)

		D_fake_error.backward()

		DQ_optimiser.step()


		# GENERATOR STEP
		G.zero_grad()
		DQ.zero_grad() # we're going to optimise it again

		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		dis_c = code.sample_discrete(batch_size)
		if gpu:
			z = z.cuda()
			dis_c = dis_c.cuda()

		z = Variable(z)
		dis_c = Variable(dis_c)
		gen_data = G(z, dis_c)
		D_gen, Q_params = DQ(gen_data, mode='Q')
		G_error = source_criterion(D_gen, ones)

		Q_logits_list = code.get_logits(Q_params)
		dis_c_list = code.get_logits(dis_c)
		Q_dis_error = 0
		for logits, onehot_targets in zip(Q_logits_list, dis_c_list):
			# apply cross-entropy loss to all different discrete random variables
			_, targets = onehot_targets.max(1) # NLL wants numerical targets, not onehot
			Q_dis_error += dis_criterion(logits, targets)
			
		Q_dis_error *= lambda_param

		G_error.backward(retain_graph=True)
		Q_dis_error.backward()

		G_optimiser.step()
		DQ_optimiser.step()


	# generates samples with fixed noise
	fake = G(fixed_z, fixed_c)
	vutils.save_image(fake.data, '{}InfoGAN1_{}_samples_epoch_{}.png'.format(SAVE_FOLDER, n_feature_maps, epoch), normalize=True, nrow=10)

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}InfoGAN1{}_generator'.format(dataset_name, n_feature_maps))
	torch.save(D.state_dict(), RESULTS_FOLDER + '{}InfoGAN1{}_D_and_Q'.format(dataset_name, n_feature_maps))