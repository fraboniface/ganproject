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

SAMPLES_FOLDER = './mnist_samples'
RESULTS_FOLDER = './results/'
N_CLASSES = 10

gpu = torch.cuda.is_available()

transform = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)

batch_size = 50
dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)

def onehot(batch,n_classes=N_CLASSES):
	ones = torch.sparse.torch.eye(n_classes)
	return ones.index_select(0,batch).view(-1,n_classes,1,1)

#custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class ConditionalGenerator(nn.Module):
    def __init__(self, zdim=100, ydim=N_CLASSES, num_features=64):
        super(ConditionalGenerator, self).__init__()
        self.deconv_noise = nn.ConvTranspose2d(zdim, num_features, 7, 1, 0, bias=False)
        self.deconv_label = nn.ConvTranspose2d(ydim, num_features, 7, 1, 0, bias=False)
        self.main = nn.Sequential(
            nn.BatchNorm2d(2*num_features),
            nn.ReLU(True),
            #7x7
            nn.ConvTranspose2d(2*num_features, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            #14x14
            nn.ConvTranspose2d(num_features, 1, 4, 2, 1, bias=False),
            #28x28 -> output
            nn.Tanh()
        )
        
    def  forward(self, z, y):
    	z = self.deconv_noise(z)
    	y = self.deconv_label(y)
    	x = torch.cat([z,y], 1)
    	return self.main(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, ydim=N_CLASSES, num_features=32):
        super(ConditionalDiscriminator, self).__init__()
        self.main = nn.Sequential(
            #28x28
            nn.Conv2d(1, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #14x14
            nn.Conv2d(num_features, 2*num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*num_features),
            nn.LeakyReLU(0.2, inplace=True),
            #7x7
        )
        self.deconv_label = nn.ConvTranspose2d(ydim, 2*num_features, 7, 1, 0, bias=False)
        self.last = nn.Conv2d(4*num_features, 1, 7, 1, 0, bias=False)
        
    def  forward(self, x, y):
    	x = self.main(x) #7x7x(2*num_features)
    	y = F.leaky_relu(self.deconv_label(y), 0.2) #idem
    	merge = torch.cat([x,y], 1)
    	# 7x7x(4*num_features)
    	logit = self.last(merge)
    	#1x1x1
    	output = F.sigmoid(logit)
    	return output.view(-1, 1).squeeze(1)


z_size = 100
G = ConditionalGenerator(z_size)
G.apply(weights_init)

D = ConditionalDiscriminator()
D.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
g_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
d_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

criterion = nn.BCELoss()

ones = 0.9*Variable(torch.ones(batch_size)) #label smoothing
zeros = Variable(torch.zeros(batch_size))

# we generate 10 samples for each class and each class has the same noise vector
n_samples_per_class  = 10
fixed_noise = Variable(torch.FloatTensor(n_samples_per_class, z_size, 1, 1).normal_(0,1))
fixed_noise = fixed_noise.repeat(N_CLASSES,1,1,1)
label_vectors = torch.FloatTensor()
for i in range(N_CLASSES):
    tmp = onehot(i*torch.ones(n_samples_per_class).long())
    label_vectors = torch.cat([label_vectors,tmp])

label_vectors = Variable(label_vectors)

# to GPU
if gpu:
	G.cuda()
	D.cuda()
	criterion.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_noise = fixed_noise.cuda()
	label_vectors = label_vectors.cuda()

samples = []
loss_d_real = []
loss_d_fake = []
loss_d = []
loss_g = []

n_epochs = 20
for epoch in tqdm(range(1,n_epochs+1)):
	for i, data in enumerate(dataloader):
		img, labels = data
		labels = onehot(labels)
		if gpu:
			img = img.cuda()
			labels = labels.cuda()

		img = Variable(img)
		labels = Variable(labels)

		# DISCRIMINATOR STEP
		D.zero_grad()

		#real data
		d_real = D(img,labels)
		d_real_error = criterion(d_real, ones)
		loss_d_real.append(d_real_error)
		d_real_error.backward()

		#fake data
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = onehot(torch.LongTensor(batch_size).random_(0,N_CLASSES))
		if gpu:
			z = z.cuda()
			y = y.cuda()
		z = Variable(z)
		y = Variable(y)
		fake_data = G(z, y)
		d_fake = D(fake_data.detach(), y)
		d_fake_error = criterion(d_fake, zeros)
		loss_d_fake.append(d_fake_error)
		d_fake_error.backward()
		loss_d.append(d_real_error + d_fake_error)

		d_optimiser.step()

		# GENERATOR STEP
		G.zero_grad()
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = onehot(torch.LongTensor(batch_size).random_(0,N_CLASSES))
		if gpu:
			z = z.cuda()
			y = y.cuda()
		z = Variable(z)
		y = Variable(y)
		gen_data = G(z, y)
		d_output = D(gen_data, y)
		g_error = criterion(d_output, ones)
		loss_g.append(g_error)
		g_error.backward()
		g_optimiser.step()

	fake = G(fixed_noise, label_vectors)
	samples.append(fake.data.cpu().numpy())
	vutils.save_image(fake.data, '%s/conditional_samples_epoch_%03d.png' % (SAMPLES_FOLDER, epoch), normalize=True)

# save everything
torch.save(G.state_dict(), RESULTS_FOLDER + 'mnist_conditional_generator_{}epochs'.format(n_epochs))
torch.save(D.state_dict(), RESULTS_FOLDER + 'mnist_conditional_discriminator_{}epochs'.format(n_epochs))
results = {
	'loss_d': loss_d,
	'loss_d_real': loss_d_real,
	'loss_d_fake': loss_d_fake,
	'loss_g': loss_g,
	'samples': samples
}

with open(RESULTS_FOLDER + 'losses_and_samples_mnist_conditionalDCGAN.p', 'wb') as f:
	pickle.dump(data, f)
