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

n_classes = 10
img_size = 32
n_channels = 1
n_feature_maps = 64
n_epochs = 20
batch_size = 50

transform = transforms.Compose(
[
	transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)

#def onehot(batch,n_classes=N_CLASSES):
#	ones = torch.sparse.torch.eye(n_classes)
#	return ones.index_select(0,batch).view(-1,n_classes,1,1)

#custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


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


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_features=n_feature_maps):
        super(ConditionalDiscriminator, self).__init__()
        #32x32
        self.conv_image = nn.Conv2d(n_channels, int(num_features/2), 4, 2, 1, bias=False)
        self.conv_label = nn.Conv2d(n_classes, int(num_features/2), 4, 2, 1, bias=False)

        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            #16x16
            nn.Conv2d(num_features, 2*num_features, 4, 2, 1, bias=False),
            #8x8
            nn.BatchNorm2d(2*num_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*num_features, 4*num_features, 4, 2, 1, bias=False),
            #4x4
            nn.BatchNorm2d(4*num_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*num_features, 1, 4, 1, 0, bias=False),
            #1x1
            nn.Sigmoid()
        )
        
    def  forward(self, x, y):
    	# y is filled to have shape batch_size x n_classes x img_size x img_size
    	x = self.conv_image(x)
    	y = self.conv_label(y)
    	z = torch.cat([x,y], 1)
    	output = self.main(z)
    	return output.view(-1, 1).squeeze(1)


z_size = 100
G = ConditionalGenerator(z_size, n_feature_maps)
G.apply(weights_init)

D = ConditionalDiscriminator(n_feature_maps)
D.apply(weights_init)

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
g_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
d_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

criterion = nn.BCELoss()

# label preprocessing
# onehot is a tensor containing the corresponding n_classesx1x1 tensor to a label 
# fill is the same but contains a  n_classes x img_size x img_size for each class
# onehot is for inputs to the generator and fill to the generator 
onehot = torch.zeros(n_classes, n_classes)
onehot = onehot.scatter_(1, torch.LongTensor(np.arange(10)).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([n_classes, n_classes, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

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
fixed_y_g = onehot[fixed_y]
#fixed_y_d = fill[fixed_y] # we don't actually use it

fixed_z = Variable(fixed_z, volatile=True)
fixed_y_g = Variable(fixed_y_g, volatile=True)
#fixed_y_d = Variable(fixed_y_d, volatile=True)

ones = 0.9*Variable(torch.ones(batch_size)) # label smoothing
zeros = Variable(torch.zeros(batch_size))

# to GPU
gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	D.cuda()
	criterion.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_z = fixed_noise.cuda()
	fixed_y_g = fixed_y_g.cuda()
	#fixed_y_d = fixed_y_d.cuda()

samples = []
loss_d_real = []
loss_d_fake = []
loss_d = []
loss_g = []

for epoch in tqdm(range(1,n_epochs+1)):
	for img, labels in dataloader:

		labels = fill[labels]
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
		y = torch.LongTensor(batch_size).random_(0,n_classes)
		y_g = onehot[y]
		y_d = fill[y]
		if gpu:
			z = z.cuda()
			y_g = y_g.cuda()
			y_d = y_d.cuda()

		z = Variable(z)
		y_g = Variable(y_g)
		y_d = Variable(y_d)
		fake_data = G(z, y_g)
		d_fake = D(fake_data.detach(), y_d)
		d_fake_error = criterion(d_fake, zeros)
		loss_d_fake.append(d_fake_error)
		d_fake_error.backward()
		loss_d.append(d_real_error + d_fake_error)

		d_optimiser.step()

		# GENERATOR STEP
		G.zero_grad()
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		y = torch.LongTensor(batch_size).random_(0,n_classes)
		y_g = onehot[y]
		y_d = fill[y]
		if gpu:
			z = z.cuda()
			y_g = y_g.cuda()
			y_d = y_d.cuda()

		z = Variable(z)
		y_g = Variable(y_g)
		y_d = Variable(y_d)
		gen_data = G(z, y_g)
		d_output = D(gen_data, y_d)
		g_error = criterion(d_output, ones)
		loss_g.append(g_error)
		g_error.backward()
		g_optimiser.step()

	fake = G(fixed_z, fixed_y_g)
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
