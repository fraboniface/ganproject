import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
from torchvision  import transforms, datasets
import torchvision.utils as vutils
from tqdm import tqdm
import sys
import pickle

from models import *
from dataset import *

model_name = 'SNWGAN2'
dataset_name = 'paintings64'
SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/regular_save/'

batch_size = 64
zdim = 100
n_feature_maps = 128
n_epochs = 100

load_model = False


print("Dataset creation...")
transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
#dataset = PaintingsDataset('../info/dataset_info.csv', '../paintings64', transform=transform)
dataset = datasets.ImageFolder('../paintings64/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print("Dataset created")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


G = Generator(zdim, n_feature_maps)
D = SNDiscriminator(n_feature_maps)
if load_model:
    G.load_state_dict(torch.load(RESULTS_FOLDER + '{}_{}_generator'.format(dataset_name, model_name), map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(RESULTS_FOLDER + '{}_{}_discriminator'.format(dataset_name, model_name), map_location=lambda storage, loc: storage))
    print("Model loaded.")
else:
    G.apply(weights_init)
    D.apply(weights_init)

fixed_z = torch.FloatTensor(batch_size, zdim, 1, 1).normal_(0,1)
fixed_z = Variable(fixed_z, volatile=True)

gpu = torch.cuda.is_available()
if gpu:
    G.cuda()
    D.cuda()
    fixed_z = fixed_z.cuda()

lr = 1e-3
beta1 = 0.5
beta2 = 0.999
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=lr, betas=(beta1, beta2))

train_hist = {
    'D_loss': [],
    'G_loss': []
    }

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
        D_err = torch.mean(D_real) - torch.mean(D_fake)
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

        D_losses.append(D_err)
        G_losses.append(G_err)  

    if epoch %5 == 0:
    # generates samples with fixed noise
        fake = G(fixed_z)
        vutils.save_image(fake.data, '{}{}_{}_samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, n_feature_maps, epoch), normalize=True, nrow=10)

        # saves everything, overwriting previous epochs
        torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_generator'.format(dataset_name, model_name, epoch))
        torch.save(D.state_dict(), RESULTS_FOLDER + '{}_{}_epoch_{}_discriminator'.format(dataset_name, model_name, epoch))

        with open(RESULTS_FOLDER + 'losses_and_samples_{}_{}_epoch_{}.p'.format(dataset_name, model_name, epoch), 'wb') as f:
            pickle.dump(results, f)