import os
import time 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import argparse
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from pathlib import Path
from tqdm import tqdm


### Parser ###
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--num_epochs', type=int, default=75, help='number of epochs')
parser.add_argument('--z_dim', type=int, default=100, help='noise dimension, default=100')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
opt = parser.parse_args()

gpu = torch.cuda.is_available()

### GENERATOR AND DISCRIMINATOR ####

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class generator(nn.Module):
    def __init__(self, z_dim=100, d=64):
        super(generator, self).__init__()
        #1x1
        self.deconv1 = nn.ConvTranspose2d(z_dim, d*16, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*16)
        #4x4
        self.deconv2 = nn.ConvTranspose2d(d*16, d*8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*8)
        #8x8
        self.deconv3 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*4)
        #16x16
        self.deconv4 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*2)
        #32x32
        self.deconv5 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d) 
        #64x64
        self.deconv6 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        #128x128
    

    def forward(self, x):
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        return F.tanh(self.deconv6(x))

class discriminator(nn.Module):
    
    def __init__(self, d=16):
        self.d = d
        super(discriminator, self).__init__()
        #128x128
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)	
        #64x64
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        #32x32
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        #16x16
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        #32x32
        self.conv5 = nn.Conv2d(d*8, d*16, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*16)
        #64x64
        self.conv6 = nn.Conv2d(d*16, 1, 4, 1, 0)
        #128x128

    def forward(self, x, matching=False):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x_intermediate = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        output = F.sigmoid(self.conv6(x_intermediate)).view(-1,1).squeeze(1)
        if matching == True:
            return output, x_intermediate
        else:
            return output

### AUXILIARY FUNCTIONS FOR PLOTTING AND SAVING RESULTS ##

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)   # fixed noise
if gpu:
    fixed_z_ = fixed_z_.cuda()
fixed_z_ = Variable(fixed_z_, volatile=True)

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    if gpu:
        z_ = z_.cuda()
    z_ = Variable(z_, volatile=True)
    
    G.eval()
    if isFix:
        test_images = G(fixed_z_.cuda())
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training param
batch_size = opt.batchSize
learning_rate = opt.lr
num_epochs = opt.num_epochs
z_size = opt.z_dim


# load and transform data
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # rerange to [-1,1]
])

portraits = datasets.ImageFolder('../all_paintings_128/', transform=transform)
train_loader = torch.utils.data.DataLoader(portraits, batch_size=batch_size, shuffle=True, num_workers=2)


# results save folder
save_folder = '../all128_dcgan_fm_results'
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
if not os.path.isdir(save_folder + '/Random_results'):
    os.mkdir(save_folder + '/Random_results')
if not os.path.isdir(save_folder + '/Fixed_results'):
    os.mkdir(save_folder + '/Fixed_results')
if not os.path.isdir(save_folder + '/Samples'):
    os.mkdir(save_folder + '/Samples')

### Model ####

G = generator()
D = discriminator()
G.apply(weights_init)
D.apply(weights_init)
lossD = nn.BCELoss()
lossG = nn.MSELoss()

G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

ones = 0.9*Variable(torch.ones(batch_size)) # Label smoothing
zeros = Variable(torch.zeros(batch_size))
fixed_noise = Variable(torch.FloatTensor(batch_size, z_size, 1, 1).normal_(0,1))

if gpu:
    G.cuda()
    D.cuda()
    lossD.cuda()
    lossG.cuda()
    ones = ones.cuda()
    zeros = zeros.cuda()
    fixed_noise = fixed_noise.cuda()
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_times'] = []
train_hist['total_time'] = []

start_time = time.time()

for epoch in tqdm(range(num_epochs)):
    # Generate samples
    fake = G(fixed_noise)
    vutils.save_image(fake.data, '%s/Samples/samples_epoch_%03d.png' % (save_folder, epoch), normalize=True)
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, _ in train_loader:
        try: # loaded dataset must divide by batch_size
            
            # Train discriminator
            D.zero_grad()
            if gpu:
                x_ = x_.cuda()
            x_ = Variable(x_)
            # Train D on real data
            D_real_result = D(x_)
            D_real_loss = lossD(D_real_result, ones) # ones=Real
            D_real_loss.backward()
            # Train D on fake data
            z_ = torch.randn((batch_size, z_size)).view(-1, z_size, 1, 1)
            if gpu:
                z_ = z_.cuda()
            z_ = Variable(z_)
            G_fake_data = G(z_)
            D_fake_result = D(G_fake_data.detach())
            D_fake_loss = lossD(D_fake_result, zeros) # zeros=Fake
            D_fake_loss.backward()
            D_optimizer.step()
            D_losses.append(D_real_loss.data[0]+D_fake_loss.data[0])

            # Train generator
            G.zero_grad()
            z_ = torch.randn((batch_size, z_size)).view(-1, z_size, 1, 1)
            if gpu:
               z_ =  z_.cuda()
            z_ = Variable(z_)
            G_fake_data = G(z_)
            ### feature matching ##
            _, feature_real = D(x_.detach(), matching=True)
            _, feature_fake = D(G_fake_data, matching=True)
            feature_real = torch.mean(feature_real,0)
            feature_fake = torch.mean(feature_fake,0)
            G_loss = lossG(feature_fake, feature_real.detach())
            ### feature matching ##
            G_loss.backward()
            G_optimizer.step()
            G_losses.append(G_loss.data[0])
        except:
            pass
            
        
    epoch_end_time = time.time()
    per_epoch_time = epoch_end_time - epoch_start_time
    
#     print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % 
#           ((epoch + 1), num_epoch, per_epoch_time, torch.mean(torch.FloatTensor(D_losses)),
#                                                               torch.mean(torch.FloatTensor(G_losses))))
    p = save_folder + '/Random_results/all_DCGAN_' + str(epoch + 1) + '.png'
    fixed_p = save_folder + '/Fixed_results/all_DCGAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_times'].append(per_epoch_time)
    
    # Saving model every epoch
    torch.save(G.state_dict(), save_folder + '/generator_param.pkl')
    torch.save(D.state_dict(), save_folder + '/discriminator_param.pkl')
    with open(save_folder + '/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

end_time = time.time()
total_time = end_time - start_time
train_hist['total_time'].append(total_time)

show_train_hist(train_hist, save=True, path=save_folder + '/all_DCGAN_train_hist.png')

# Create GIF
images = []
for e in range(num_epochs):
    img_name = save_folder + '/Fixed_results/all_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(save_folder + '/generation_animation.gif', images, fps=5)
