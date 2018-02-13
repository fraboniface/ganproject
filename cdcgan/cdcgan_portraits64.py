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

# THIS IS A WGAN

SAVE_FOLDER = '../results/samples/paintings/portraits64/'
RESULTS_FOLDER = '../saved_data/'

n_classes = 21
img_size = 64
n_channels = 3
n_feature_maps = 128
n_epochs = 200
batch_size = 100
D_steps = 5
clamp = 1e-2

transform = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
portraits = datasets.ImageFolder('../paintings64/portraits/', transform=transform)
dataloader = torch.utils.data.DataLoader(portraits, batch_size=batch_size, shuffle=True, num_workers=2)


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


class WConditionalDiscriminator(nn.Module):
    def __init__(self, num_features=n_feature_maps):
        super(WConditionalDiscriminator, self).__init__()
        #32x32
        self.conv_image = nn.Conv2d(n_channels, int(num_features/2), 4, 2, 1, bias=False)
        self.conv_label = nn.Conv2d(n_classes, int(num_features/2), 4, 2, 1, bias=False)
        self.main = nn.Sequential(
            #64x64
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
            nn.Conv2d(8*n_feature_maps, 1, 4, 1, 0, bias=False),
            #1x1
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

D = WConditionalDiscriminator(n_feature_maps)
D.apply(weights_init)

lr = 5e-5
g_optimiser = optim.RMSprop(G.parameters(),lr=lr)
d_optimiser = optim.RMSprop(G.parameters(),lr=lr)

# label preprocessing
# onehot is a tensor containing the corresponding n_classesx1x1 tensor to a label 
# fill is the same but contains a  n_classes x img_size x img_size for each class
# onehot is for inputs to the generator and fill to the generator 
onehot = torch.zeros(n_classes, n_classes)
onehot = onehot.scatter_(1, torch.LongTensor(np.arange(n_classes)).view(n_classes,1), 1).view(n_classes, n_classes, 1, 1)
fill = torch.zeros([n_classes, n_classes, img_size, img_size])
for i in range(n_classes):
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
	fixed_z = fixed_z.cuda()
	fixed_y_g = fixed_y_g.cuda()
	#fixed_y_d = fixed_y_d.cuda()

samples = []
loss_d = []
loss_g = []

for epoch in tqdm(range(1,n_epochs+1)):
	data_iter = iter(dataloader)
	i = 0
	while i < len(dataloader):
		# insert code where D_steps is set to 100 once in a while, if needed

		# DISCRIMINATOR TRAINING
		j = 0
		while j < D_steps and i < len(dataloader):
			j += 1

			for p in D.parameters():
				p.data.clamp_(-clamp, clamp)

			img, labels = data_iter.next()
			labels = fill[labels]
			if gpu:
				img = img.cuda()
				labels = labels.cuda()

			img = Variable(img)
			labels = Variable(labels)

			D.zero_grad()

			#real data
			d_real = D(img,labels)
			d_real_mean = torch.mean(d_real)

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
			d_fake_mean = torch.mean(d_fake)

			d_error = -(d_real_mean - d_fake_mean)

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
		g_error = -torch.mean(d_output)
		g_error.backward()
		g_optimiser.step()


		loss_d.append(d_error.data.cpu().numpy())
		loss_g.append(g_error.data.cpu().numpy())

	fake = G(fixed_z, fixed_y_g)
	samples.append(fake.data.cpu().numpy())
	vutils.save_image(fake.data, '{}cWGAN_samples_epoch_{}.png'.format(SAVE_FOLDER, epoch), normalize=True, nrow=10)

# save everything
torch.save(G.state_dict(), RESULTS_FOLDER + 'portraits64_cWGAN_generator_{}epochs'.format(n_epochs))
torch.save(D.state_dict(), RESULTS_FOLDER + 'portraits64_cWGAN_discriminator_{}epochs'.format(n_epochs))
results = {
	'D_loss': loss_d,
	'G_loss': loss_g,
	'samples': samples
}
with open(RESULTS_FOLDER + 'losses_and_samples_portraits64_cWGAN.p', 'wb') as f:
	pickle.dump(results, f)