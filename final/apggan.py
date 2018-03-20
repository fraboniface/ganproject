import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
from torchvision  import transforms, datasets
import torchvision.utils as vutils
from tqdm import tqdm
import pickle

from models import *
from dataset import *

model_name = 'APG_SNGAN'
dataset_name = 'paintings128'
SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'

batch_size = 64
zdim = 100
n_feature_maps = 128
init_size = 4
final_size = 128
n_epochs = 100

epsilon_drift = 1e-3
n_samples_seen = 1e5

print("Dataset creation...")

transform = transforms.Compose(
	[
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
	])
#dataset = PaintingsDataset('../info/dataset_info.csv', '../paintings64', transform=transform)
dataset = datasets.ImageFolder('../paintings128/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print("Dataset created")

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv' or 'SNConv') != -1:
		nn.init.kaiming_normal(m.weight.data)

G = AbruptGrowingGenerator(zdim, init_size, final_size, n_feature_maps)
G.apply(weights_init)
D = AbruptGrowingDiscriminator(init_size, final_size, n_feature_maps)
D.apply(weights_init)

fixed_z = torch.FloatTensor(batch_size, zdim, 1, 1).normal_(0,1)
fixed_z = Variable(fixed_z, volatile=True)

gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	D.cuda()
	fixed_z = fixed_z.cuda()

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=lr, betas=(beta1, beta2))

train_hist = {
	'D_loss': [],
	'G_loss': []
	}

examples_seen = 0
current_size = init_size
for epoch in tqdm(range(1,n_epochs+1)):
	i = 0
	D_losses = []
	G_losses = []
	for x, label in dataloader:
		i += 1
		print('{}: Epoch {}, batch {} starting...'.format(model_name, epoch, i))
		if gpu:
			x = x.cuda()

		x = Variable(x)
		if x.size(-1) > current_size:
			ratio = int(x.size(-1)/current_size)
			x = F.avg_pool2d(x, ratio)

		# D training, n_critic=1
		for p in D.parameters():
			p.requires_grad = True

		D.zero_grad()
		D_real = D(x)

		z = torch.FloatTensor(x.size(0), zdim, 1, 1).normal_()
		if gpu:
			z = z.cuda()

		z = Variable(z)
		fake = G(z)
		D_fake = D(fake.detach())        
		D_err = torch.mean(D_real) - torch.mean(D_fake) + epsilon_drift*torch.mean(D_real**2)
		D_err.backward()
		D_optimiser.step()

		# G training
		for p in D.parameters():
			p.requires_grad = False # saves computation

		G.zero_grad()

		z = torch.FloatTensor(batch_size, zdim, 1, 1).normal_()
		if gpu:
			z = z.cuda()

		z = Variable(z)
		fake = G(z)
		G_err = torch.mean(D(fake))
		G_err.backward()
		G_optimiser.step()

		examples_seen += x.size(0)

		D_losses.append(D_err)
		G_losses.append(G_err)

	# we grow every n_samples_seen images (more or less bacause we wait for the end of the epoch anyway)
	if examples_seen > n_samples_seen:
		examples_seen = 0
		current_size *= 2
		G.grow()
		D.grow()
		if gpu:
			G.cuda()
			D.cuda()
			
		G_optimiser.add_param_group({'params': G.new_parameters})
		D_optimiser.add_param_group({'params': filter(lambda p: p.requires_grad, D.new_parameters)})
		print('Networks grown, current size is', current_size)

			

	# generates samples with fixed noise
	fake = G(fixed_z)
	vutils.save_image(fake.data, '{}{}__samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, epoch), normalize=True, nrow=10)

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_generator'.format(dataset_name, model_name))
	torch.save(D.state_dict(), RESULTS_FOLDER + '{}_{}_discriminator'.format(dataset_name, model_name))

	train_hist['D_loss'].append(np.array(D_losses).mean())
	train_hist['G_loss'].append(np.array(G_losses).mean())
	with open(RESULTS_FOLDER + 'losses_{}_{}.p'.format(dataset_name, model_name), 'wb') as f:
		pickle.dump(train_hist, f)