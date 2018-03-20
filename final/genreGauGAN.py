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
from dataset import *

model_name = 'InfoACGauGAN_genre'
dataset_name = 'paintings64'
SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'


# **********************HYPER-PARAMS (EXCEPT LEARNING RATE)*************************

batch_size = 50
img_size = 64
n_feature_maps = 128
n_epochs = 60

INFO = True
if INFO:
	n_info_vars = 2
	lambda_ = 0.5

# *****************************DATA HANDLING**************************************
print("Dataset creation")
transform = transforms.Compose(
[
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
#dataset = PaintingsDataset('../info/dataset_info.csv', '../paintings64', transform=transform)
dataset = datasets.ImageFolder('../paintings64/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

n_genre_classes = len(dataset.classes)

print("Dataset created, containing {} samples.".format(len(dataset)))


# ********************************MODEL CREATION*******************************

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

z_size = 100 # the last n_info_vars are the code, with the same prior as the rest of z
G = Generator(z_size + n_genre_classes, n_feature_maps)
G.apply(weights_init)

if INFO:
	D = Q_ACDiscriminator_1class(n_genre_classes, n_feature_maps, n_info_vars)
else:
	D = Q_ACDiscriminator_1class(n_genre_classes, n_feature_maps)
D.apply(weights_init)

print("Model created.")

# ************************************LOSSES**************************************

source_criterion = nn.BCELoss()
# targets
#ones = Variable(torch.ones(batch_size))
ones = Variable(torch.FloatTensor(batch_size).uniform_(0.9,1)) # label smoothing
zeros = Variable(torch.zeros(batch_size))

genre_weights = np.array([4089,10983,11545,12926,5702], dtype=np.float)
genre_weights = 1 / torch.Tensor(genre_weights)
genre_criterion = nn.CrossEntropyLoss(genre_weights)


feature_matching_criterion = nn.MSELoss()

if INFO:
	def log_gaussian(mean, var, x):
		"""Arguments are vectors"""
		log = -1/2 * torch.log(2*np.pi*var) - (x-mean)**2 / (2*var)
		return log.sum(1).mean()


# **********************FIXED INPUT FOR MONITORING PROGRESS ON GENERATION********************

onehot_genre = torch.eye(n_genre_classes).view(n_genre_classes,n_genre_classes,1,1)

# we generate 10 samples for each class and each class has the same noise vector
n_samples_per_class  = 10
fixed_z = torch.FloatTensor(n_samples_per_class, z_size, 1, 1).normal_(0,1)
fixed_z = fixed_z.repeat(n_genre_classes,1,1,1)

# genre changes while style remains impressionism
fixed_genre = []
for i in range(n_genre_classes):
	fixed_genre += [i]*n_samples_per_class
fixed_genre = torch.LongTensor(fixed_genre)
fixed_genre = onehot_genre[fixed_genre]

fixed_z = torch.cat([fixed_z,fixed_genre],1)
fixed_z = Variable(fixed_z, volatile=True)


# ************************************GPU HANDLING*********************************

gpu = torch.cuda.is_available()
if gpu:
	G.cuda()
	D.cuda()
	source_criterion.cuda()
	genre_criterion.cuda()
	feature_matching_criterion.cuda()
	ones = ones.cuda()
	zeros = zeros.cuda()
	fixed_z = fixed_z.cuda()


# *************************************OPTIMISERS*********************************

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

print("Optimisers created.")


# ***********************************TRAINING HISTORY**************************************

train_hist = {
	'D_loss': [],
	'G_loss': []
}
if INFO:
	train_hist['milb'] = []

# ************************************TRAINING LOOP*********************************************

for epoch in tqdm(range(1,n_epochs+1)):
	print("Epoch starting...")
	i = 0
	D_losses = []
	G_losses = []
	if INFO:
		milbs = []

	for img, genres in dataloader:
		i += 1
		print('Epoch {}, batch {} starting...'.format(epoch, i))
		if img.size(0) < batch_size:
			continue
		if gpu:
			img = img.cuda()
			genres = genres.cuda()

		x = Variable(img)
		genres = Variable(genres, volatile=True)


		# *************************************DISCRIMINATOR STEP********************************

		D.zero_grad()

		# ******************* REAL DATA ********************
		if INFO:
			D_real_source, D_real_genre = D(x, mode='D', input_source='real')
		else:
			D_real_source, D_real_genre = D(x)

		D_real_Ls = source_criterion(D_real_source, ones)
		D_real_Lc_genre = genre_criterion(D_real_genre, genres)
		D_real_error = D_real_Ls + D_real_Lc_genre

		#******************** FAKE DATA ***********************
		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		genre = torch.LongTensor(batch_size).random_(0, n_genre_classes)
		genre_oh = onehot_genre[genre]
		z = torch.cat([z, genre_oh], 1)
		if gpu:
			z = z.cuda()
			genre= genre.cuda()

		z = Variable(z)
		genre = Variable(genre)
		fake_data = G(z)
		if INFO:
			D_fake_source, D_fake_genre, mean_c_x, var_c_x = D(fake_data.detach(), mode='D', input_source='fake')
		else:
			D_fake_source, D_fake_genre = D(fake_data.detach())

		D_fake_Ls = source_criterion(D_fake_source, zeros)
		D_fake_Lc_genre = genre_criterion(D_fake_genre, genre)
		D_fake_error = D_fake_Ls + D_fake_Lc_genre

		D_loss = D_real_error + D_fake_error

		if INFO:
			Q_obj = log_gaussian(z[-1,-n_info_vars:,:,:], mean_c_x, var_c_x)
			D_loss -= lambda_*Q_obj

		# *******ERROR BACKPROP AND OPTIMISER STEP***************
		D_loss.backward()

		D_optimiser.step()


		# **************************************GENERATOR STEP****************************************
		G.zero_grad()

		z = torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1)
		genre = torch.LongTensor(batch_size).random_(0, n_genre_classes)
		genre_oh = onehot_genre[genre]
		z = torch.cat([z, genre_oh], 1)
		if gpu:
			z = z.cuda()
			genre= genre.cuda()

		z = Variable(z)
		genre = Variable(genre)
		fake_data = G(z)
		real_features = D(x, only_fm=True)

		if INFO:
			fake_features, D_fake_genre, mean_c_x, var_c_x = D(fake_data.detach(), mode='G', input_source='fake')
		else:
			fake_features, D_fake_genre = D(fake_data, fm=True)

		real_mean = torch.mean(real_features, 0)
		fake_mean = torch.mean(fake_features, 0)
		G_error = feature_matching_criterion(fake_mean, real_mean.detach())
		DG_Lc_genre = genre_criterion(D_fake_genre, genre)
		DG_error = DG_Lc_genre

		G_loss = G_error + DG_error

		if INFO:
			QG_obj = log_gaussian(z[-1,-n_info_vars:,:,:], mean_c_x, var_c_x)
			G_loss -= lambda_*QG_obj

		G_loss.backward()
		G_optimiser.step()

		D_losses.append(D_loss)
		G_losses.append(G_losses)
		if INFO:
			milbs.append(QG_obj)


	# ***************************** SAVE SAMPLES AND LOSSES AFTER EACH EPOCH *******************************

	fake = G(fixed_z)
	vutils.save_image(fake.data, '{}{}__samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, epoch), normalize=True, nrow=n_samples_per_class)
	print("Samples saved")

	# saves everything, overwriting previous epochs
	torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_generator'.format(dataset_name, model_name))
	torch.save(D.state_dict(), RESULTS_FOLDER + '{}_{}_discriminator'.format(dataset_name, model_name))
	print("Model saved")

	train_hist['D_loss'].append(np.array(D_losses).mean())
	train_hist['G_loss'].append(np.array(G_losses).mean())
	if INFO:
		train_hist['milb'].append(np.array(milbs).mean())

	with open(RESULTS_FOLDER + 'losses_{}_{}.p'.format(dataset_name, model_name), 'wb') as f:
	    pickle.dump(train_hist, f)
	print("Losses saved")