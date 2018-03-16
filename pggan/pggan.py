import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
from torchvision  import transforms, datasets
import torchvision.utils as vutils
from tqdm import tqdm
from models import *

model_name = 'PG_GAN'
dataset_name = 'paintings64'
SAVE_FOLDER = '../results/samples/{}/'.format(dataset_name)
RESULTS_FOLDER = '../results/saved_data/'

batch_size = 64
zdim = 100
n_feature_maps = 128
init_size = 4
final_size = 64
n_epochs = 50

lambda_ = 10
gamma = 750
epsilon_drift = 1e-3
n_samples_seen = 5e5

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
dataset = datasets.ImageFolder('../paintings64/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data)
        
G = GrowingGenerator(zdim, init_size, final_size, n_feature_maps)
G.apply(weights_init)
D = GrowingDiscriminator(init_size, final_size, n_feature_maps)
D.apply(weights_init)

def get_gradient_penalty(real, fake, D, gamma=1, gpu=True):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size,1,1,1)
    alpha = Variable(alpha.expand_as(real))
    if gpu:
        alpha = alpha.cuda()

    interpolation = alpha * real + (1-alpha) * fake # everything is a Variable so interpolation should be one too
    D_itp = D(interpolation)
    if gpu:
        gradients = grad(outputs=D_itp, inputs=interpolation, grad_outputs=torch.ones(D_itp.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = grad(outputs=D_itp, inputs=interpolation, grad_outputs=torch.ones(D_itp.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]

    GP = ((gradients.norm(2, dim=1) - gamma)**2 / gamma**2).mean()
    return GP

fixed_z = torch.FloatTensor(batch_size, zdim, 1, 1).normal_(0,1)
fixed_z = Variable(fixed_z, volatile=True)

gpu = torch.cuda.is_available()
if gpu:
    G.cuda()
    D.cuda()
    fixed_z = fixed_z.cuda()

lr = 1e-3
beta1 = 0
beta2 = 0.99
G_optimiser = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
D_optimiser = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

examples_seen = 0
current_size = 4
for epoch in tqdm(range(1,n_epochs+1)):
    for img, label in dataloader:
        print('something')
        if gpu:
            img = img.cuda()

        x = Variable(img)
        if x.size(-1) > current_size:
            ratio = int(x.size(0)/current_size)
            x = F.avg_pool2d(x, ratio)
        
        # D training, n_critic=1
        for p in D.parameters():
            p.requires_grad = True
            
        D.zero_grad
        D_real = D(x)
        
        z = torch.FloatTensor(x.size(0), zdim, 1, 1).normal_()
        if gpu:
            z = z.cuda()

        z = Variable(z)
        fake = G(z)
        D_fake = D(fake.detach())
        
        GP = get_gradient_penalty(x, fake, D, gamma, gpu)
        
        D_err = torch.mean(D_real) - torch.mean(D_fake) + lambda_*GP + epsilon_drift*torch.mean(D_real**2)
        D_optimiser.step()
        
        # G training
        for p in D.parameters():
            p.requires_grad = False # saves computation
            
        z = torch.FloatTensor(batch_size, zdim, 1, 1).normal_()
        if gpu:
            z = z.cuda()
        
        z = Variable(z)
        fake = G(z)
        G_err = torch.mean(D(fake))
        G_optimiser.step()
        
        examples_seen += img.size(0)
    
    # we grow every 100K images. 600Kin the paper, plus transitions, we'll see
    if examples_seen % n_samples_seen == 0:
        examples_seen = 0
        current_size *= 2
        G.grow()
        G_optimiser.add_param_group({'params': G.new_parameters})
        D.grow()
        D_optimiser.add_param_group({'params': D.new_parameters})

    # generates samples with fixed noise
    fake = G(fixed_z)
    vutils.save_image(fake.data, '{}{}__samples_epoch_{}.png'.format(SAVE_FOLDER, model_name, epoch), normalize=True, nrow=10)

    # saves everything, overwriting previous epochs
    torch.save(G.state_dict(), RESULTS_FOLDER + '{}_{}_generator'.format(dataset_name, model_name))
    torch.save(D.state_dict(), RESULTS_FOLDER + '{}_{}_discriminator'.format(dataset_name, model_name))
