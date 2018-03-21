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

model_name = 'ACGAN3'

dataset_name = sys.argv[1]
assert dataset_name in ['paintings64', 'mnist', 'cifar']

SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/regular_save/'

batch_size = 100

if dataset_name == 'paintings64':
	n_classes = 5
	img_size = 64
	n_channels = 3
	n_feature_maps = 128
	n_epochs = 60

elif dataset_name == 'cifar':
	n_classes = 10
	img_size = 32
	n_channels = 3
	n_feature_maps = 128
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


if dataset_name == 'paintings64':
	dataset = datasets.ImageFolder('../paintings64/', transform=transform)
elif dataset_name == 'cifar':
	dataset = datasets.CIFAR10('../data', train=True, transform=transform)
elif dataset_name == 'mnist':
	dataset = datasets.MNIST('../data', train=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(dataset.classes)

#custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

if dataset_name == 'paintings64':
	n_examples = len(dataset)
	class_weights = np.array([4089,10983,11545,12926,5702], dtype=np.float) # not ideal hard-coded like this
	class_weights = 1/torch.Tensor(class_weights)
	class_criterion = nn.CrossEntropyLoss(class_weights)

	class ACGenerator(nn.Module):
	    def __init__(self, zdim=100, n_feature_maps=64):
	        super(ACGenerator, self).__init__()
	        self.main = nn.Sequential(
	        	#1x1
	        	nn.ConvTranspose2d(zdim+n_classes, 8*n_feature_maps, 4, 1, 0, bias=False),
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
	            nn.Conv2d(8*n_feature_maps, n_feature_maps, 4, 1, 0, bias=False)
	            #1x1
	        )
	        self.source = nn.Sequential(
	        	nn.Conv2d(n_feature_maps, 1, 1, 1, 0, bias=False),
	        	nn.Sigmoid()
	        )
	        self.class_logits = nn.Conv2d(n_feature_maps, n_classes, 1, 1, 0, bias=False)

	    def  forward(self, x, matching=False):
	        x = self.main(x)

	        class_logits = self.class_logits(x).view(-1, n_classes)
	        if matching :
	            return x, class_logits
	        else:
	            source = self.source(x).view(-1, 1).squeeze(1)
	            return x, source, class_logits

else:
	class_criterion = nn.CrossEntropyLoss()

	class ACGenerator(nn.Module):
	    def __init__(self, zdim=100, n_feature_maps=64):
	        super(ACGenerator, self).__init__()
	        self.main = nn.Sequential(
	        	#1x1
	        	nn.ConvTranspose2d(zdim+n_classes, 8*n_feature_maps, 4, 1, 0, bias=False),
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
	            nn.Conv2d(4*n_feature_maps, n_feature_maps, 4, 1, 0, bias=False)
	            #1x1
	        )
	        self.source = nn.Sequential(
	        	nn.Conv2d(n_feature_maps, 1, 1, 1, 0, bias=False),
	        	nn.Sigmoid()
	        )
	        self.class_logits = nn.Conv2d(n_feature_maps, n_classes, 1, 1, 0, bias=False)

	    def  forward(self, x):
	    	x = self.main(x)
	    	source = self.source(x).view(-1, 1).squeeze(1)
	    	class_logits = self.class_logits(x).view(-1, n_classes)
	    	return source, class_logits


z_size = 100
G = ACGenerator(z_size, n_feature_maps)
G.apply(weights_init)

D = ACDiscriminator(n_feature_maps)
D.apply(weights_init)

# label preprocessing
# tensor containing the corresponding n_classesx1x1 tensor for each label 
onehot = torch.eye(n_classes).view(n_classes,n_classes,1,1)

# we generate 10 samples for each class and each class has the same noise vector
n_samples_per_class  = 10
fixed_z = torch.FloatTensor(n_samples_per_class, z_size, 1, 1).normal_(0,1)
fixed_z = fixed_z.repeat(n_classes,1,1,1)
fixed_y = []
for i in range(n_classes):
	fixed_y += [i]*n_samples_per_class
fixed_y = torch.LongTensor(fixed_y)
fixed_y = onehot[fixed_y]

fixed_z = torch.cat([fixed_z,fixed_y],1)
fixed_z = Variable(fixed_z, volatile=True)

ones = Variable(torch.FloatTensor(batch_size).uniform_(0.9,1))
zeros = Variable(torch.zeros(batch_size))

G_criterion = nn.MSELoss()
D_criterion = nn.BCELoss()

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	D.cuda()
	D_criterion.cuda()
	class_criterion.cuda()
	G_criterion.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_z = fixed_z.cuda()

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

train_hist = {
	'D_loss': [],
	'G_loss': []
}

for epoch in tqdm(range(1,n_epochs+1)):
	D_losses = []
	G_losses = []
	for img, labels in dataloader:
		if img.size(0) < batch_size:
			continue
		if gpu:
			img = img.cuda()
			labels = labels.cuda()

		img = Variable(img)
		labels = Variable(labels)

		# DISCRIMINATOR STEP
		D.zero_grad()

		#real data
		D_real_features, D_real_source, D_real_class = D(img)
		D_real_Ls = D_criterion(D_real_source, ones)
		D_real_Lc = class_criterion(D_real_class, labels)
		D_real_error = D_real_Ls + D_real_Lc

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = torch.LongTensor(batch_size).random_(0,n_classes)
		y_g = onehot[y]
		z = torch.cat([z,y_g],1)
		if gpu:
			z = z.cuda()
			y = y.cuda()

		z = Variable(z)
		y = Variable(y)
		fake_data = G(z)
		_, D_fake_source, D_fake_class = D(fake_data.detach())
		D_fake_Ls = D_criterion(D_fake_source, zeros)
		D_fake_Lc = class_criterion(D_fake_class, y)
		D_fake_error = D_fake_Ls + D_fake_Lc
		
		D_loss = D_real_error + D_fake_error
		D_loss.backward()
		D_optimiser.step()


		# GENERATOR STEP
		G.zero_grad()

		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = torch.LongTensor(batch_size).random_(0,n_classes)
		y_g = onehot[y]
		z = torch.cat([z,y_g],1)
		if gpu:
			z = z.cuda()
			y = y.cuda()

		z = Variable(z)
		y = Variable(y)
		gen_data = G(z)
		D_fake_features, D_gen_class = D(gen_data, matching=True)
		D_real_features = torch.mean(D_real_features, 0)
		D_fake_features = torch.mean(D_fake_features,0)
		gen_Ls = G_criterion(D_fake_features, D_real_features.detach())
		gen_Lc = class_criterion(D_gen_class, y)
		G_loss = gen_Ls + gen_Lc

		G_loss.backward()
		G_optimiser.step()

		D_losses.append(D_loss.data[0])
		G_losses.append(G_loss.data[0])


	if epoch % 5 == 0:

		# generates samples with fixed noise
		fake = G(fixed_z)
		vutils.save_image(fake.data, '{}{}_samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, epoch), normalize=True, nrow=10)

		train_hist['D_loss'].append(torch.mean(torch.FloatTensor(D_losses)))
		train_hist['G_loss'].append(torch.mean(torch.FloatTensor(G_losses)))

		with open(RESULTS_FOLDER + 'losses_{}_{}_epoch_{}.p'.format(dataset_name, model_name, epoch), 'wb') as f:
			pickle.dump(train_hist, f)

		# saves everything, overwriting previous epochs
		torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_generator'.format(dataset_name, model_name, epoch))
		torch.save(DQ.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_D_and_Q'.format(dataset_name, model_name, epoch))