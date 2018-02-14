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

dataset_name = sys.argv[1]
assert dataset_name in ['portraits64', 'mnist', 'cifar']

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'

batch_size = 100

if dataset_name == 'portraits64':
	n_classes = 21
	img_size = 64
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 70

elif dataset_name == 'cifar':
	n_classes = 10
	img_size = 32
	n_channels = 3
	n_feature_maps = 64
	n_epochs = 50
	
elif dataset_name == 'mnist':
	n_classes = 10
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


if dataset_name == 'portraits64':
	dataset = datasets.ImageFolder('../paintings64/portraits/', transform=transform)
elif dataset_name == 'cifar':
	dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
elif dataset_name == 'mnist':
	dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


#custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

if dataset_name == 'portraits64':
	class ConditionalGenerator(nn.Module):
	    def __init__(self, zdim=100, num_features=n_feature_maps):
	        super(ConditionalGenerator, self).__init__()
	        self.deconv_noise = nn.ConvTranspose2d(zdim, 4*num_features, 4, 1, 0, bias=False)
	        self.deconv_label = nn.ConvTranspose2d(n_classes, 4*num_features, 4, 1, 0, bias=False)
	        self.main = nn.Sequential(
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
	            nn.ConvTranspose2d(n_feature_maps, 3, 4, 2, 1, bias=False),
	            #64x64
	            nn.Tanh()
	        )
	        
	    def  forward(self, z, y):
	    	z = self.deconv_noise(z)
	    	y = self.deconv_label(y)
	    	x = torch.cat([z,y], 1)
	    	# same to apply batch norm and relu before or after concatenating
	    	return self.main(x)


	class ACDiscriminator(nn.Module):
	    def __init__(self, num_features=n_feature_maps):
	        super(ACDiscriminator, self).__init__()
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
	        self.output_source = nn.Sequential(
	        	nn.Conv2d(8*n_feature_maps, 1, 4, 1, 0, bias=False),
	        	nn.Sigmoid()
	        )
	        self.class_logits = nn.Conv2d(8*n_feature_maps, n_classes, 4, 1, 0, bias=False)

	    def  forward(self, x):
	    	x = self.main(x)
	    	output_source = self.output_source(x).view(-1, 1).squeeze(1)
	    	class_logits = self.class_logits(x).view(-1, n_classes)
	    	return output_source, class_logits

else:
	class ConditionalGenerator(nn.Module):
	    def __init__(self, zdim=100, num_features=n_feature_maps):
	        super(ConditionalGenerator, self).__init__()
	        self.deconv_noise = nn.ConvTranspose2d(zdim, 2*num_features, 4, 1, 0, bias=False)
	        self.deconv_label = nn.ConvTranspose2d(n_classes, 2*num_features, 4, 1, 0, bias=False)
	        self.main = nn.Sequential(
	        	#4x4
	            nn.BatchNorm2d(4*num_features),
	            nn.ReLU(True),
	            nn.ConvTranspose2d(4*num_features,2*num_features, 4, 2, 1, bias=False),
	            #8x8
	            nn.BatchNorm2d(2*num_features),
	            nn.ReLU(True),
	            nn.ConvTranspose2d(2*num_features, num_features, 4, 2, 1, bias=False),
	            #16x16
	            nn.BatchNorm2d(num_features),
	            nn.ReLU(True),
	            nn.ConvTranspose2d(num_features, n_channels, 4, 2, 1, bias=False),
	            #32x32
	            nn.Tanh()
	        )
	        
	    def  forward(self, z, y):
	    	z = self.deconv_noise(z)
	    	y = self.deconv_label(y)
	    	x = torch.cat([z,y], 1)
	    	# same to apply batch norm and relu before or after concatenating
	    	return self.main(x)


	class ACDiscriminator(nn.Module):
	    def __init__(self, num_features=n_feature_maps):
	        super(ACDiscriminator, self).__init__()
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
	        self.output_source = nn.Sequential(
	        	nn.Conv2d(4*n_feature_maps, 1, 4, 1, 0, bias=False),
	        	nn.Sigmoid()
	        )
	        self.class_logits = nn.Conv2d(4*n_feature_maps, n_classes, 4, 1, 0, bias=False)

	    def  forward(self, x):
	    	x = self.main(x)
	    	output_source = self.output_source(x).view(-1, 1).squeeze(1)
	    	class_logits = self.class_logits(x).view(-1, n_classes)
	    	return output_source, class_logits


z_size = 100
G = ConditionalGenerator(z_size, n_feature_maps)
G.apply(weights_init)

D = ACDiscriminator(n_feature_maps)
D.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

# label preprocessing
# tensor containing the corresponding n_classesx1x1 tensor to a label 
onehot = torch.zeros(n_classes, n_classes)
onehot = onehot.scatter_(1, torch.LongTensor(np.arange(n_classes)).view(n_classes,1), 1).view(n_classes, n_classes, 1, 1)

# we generate 10 samples for each class and each class has the same noise vector
n_samples_per_class  = 10
fixed_z = torch.FloatTensor(n_samples_per_class, z_size, 1, 1).normal_(0,1)
fixed_z = fixed_z.repeat(n_classes,1,1,1)
fixed_y = torch.FloatTensor()
for i in range(n_classes):
    tmp = onehot[i*torch.ones(n_samples_per_class).long()]
    fixed_y = torch.cat([fixed_y,tmp])

fixed_y = []
for i in range(n_classes):
	fixed_y += [i]*n_samples_per_class
fixed_y = torch.LongTensor(fixed_y)
fixed_y = onehot[fixed_y]

fixed_z = Variable(fixed_z, volatile=True)
fixed_y = Variable(fixed_y, volatile=True)

ones = 0.9*Variable(torch.ones(batch_size)) # label smoothing
zeros = Variable(torch.zeros(batch_size))

source_criterion = nn.BCELoss()
class_criterion = nn.CrossEntropyLoss()

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	D.cuda()
	source_criterion.cuda()
	class_criterion.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_z = fixed_z.cuda()
	fixed_y = fixed_y.cuda()

results = {
	'samples': [],
	'real_Ls': [],
	'real_Lc': [],
	'D_real_loss': [],
	'fake_Ls': [],
	'fake_Lc': [],
	'D_fake_loss': [],
	'gen_Ls': [],
	'gen_Lc': [],
	'G_loss': []

}

for epoch in tqdm(range(1,n_epochs+1)):
	for img, labels in dataloader:
		if gpu:
			img = img.cuda()
			labels = labels.cuda()

		img = Variable(img)
		labels = Variable(labels)

		# DISCRIMINATOR STEP
		D.zero_grad()

		#real data
		D_real_source, D_real_class = D(img)
		D_real_Ls = source_criterion(D_real_source, ones)
		results['real_Ls'].append(D_real_Ls)
		D_real_Lc = class_criterion(D_real_class, labels)
		results['real_Lc'].append(D_real_Lc)
		D_real_error = D_real_Ls + D_real_Lc
		results['D_real_loss'].append(D_real_error)
		D_real_error.backward()

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = torch.LongTensor(batch_size).random_(0,n_classes)
		y_g = onehot[y]
		if gpu:
			z = z.cuda()
			y = y.cuda()
			y_g = y_g.cuda()

		z = Variable(z)
		y = Variable(y)
		y_g = Variable(y_g)
		fake_data = G(z, y_g).detach()
		D_fake_source, D_fake_class = D(fake_data)
		D_fake_Ls = source_criterion(D_fake_source, zeros)
		results['fake_Ls'].append(D_fake_Ls.data.cpu().numpy())
		D_fake_Lc = class_criterion(D_fake_class, y)
		results['fake_Lc'].append(D_fake_Lc.data.cpu().numpy())
		D_fake_error = D_fake_Ls + D_fake_Lc
		results['D_fake_loss'].append(D_fake_error.data.cpu().numpy())
		D_fake_error.backward()

		D_optimiser.step()

		# GENERATOR STEP
		G.zero_grad()

		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = torch.LongTensor(batch_size).random_(0,n_classes)
		y_g = onehot[y]
		if gpu:
			z = z.cuda()
			y = y.cuda()
			y_g = y_g.cuda()

		z = Variable(z)
		y = Variable(y)
		y_g = Variable(y_g)
		gen_data = G(z, y_g)
		D_gen_source, D_gen_class = D(gen_data)
		gen_Ls = source_criterion(D_gen_source, ones)
		results['gen_Ls'].append(gen_Ls.data.cpu().numpy())
		gen_Lc = class_criterion(D_gen_class, y)
		results['gen_Lc'].append(gen_Lc.data.cpu().numpy())
		G_error = gen_Lc - gen_Ls
		results['G_loss'].append(G_error.data.cpu().numpy())
		G_error.backward()

		G_optimiser.step()


	# generates samples with fixed noise
	fake = G(fixed_z, fixed_y)
	results['samples'].append(fake.data.cpu().numpy())
	vutils.save_image(fake.data, '{}ACGAN_samples_epoch_{}.png'.format(SAVE_FOLDER, epoch), normalize=True, nrow=n_samples_per_class)

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}_ACGAN_generator'.format(dataset_name))
	torch.save(D.state_dict(), RESULTS_FOLDER + '{}_ACGAN_discriminator'.format(dataset_name))

	with open(RESULTS_FOLDER + 'losses_and_samples_{}_ACGAN.p'.format(dataset_name), 'wb') as f:
		pickle.dump(results, f)