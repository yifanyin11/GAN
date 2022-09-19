import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lr_g", type=float, default=0.001, help="adam: learning rate for generator")
parser.add_argument("--lr_d", type=float, default=0.0001, help="adam: learning rate for discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--g_input_size", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=600, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

init_feat_num = 128

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
class ResBlock(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
    
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_feat, in_feat, 3),
            nn.InstanceNorm2d(in_feat),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = x + self.block(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = opt.img_size // 8
        # First layer to convert a random vector to a feature map
        self.linear = nn.Sequential(
            nn.Linear(opt.g_input_size, 1500),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm1d(1500, 0.8),
            nn.Linear(1500, init_feat_num*self.init_size**2),
            nn.LeakyReLU(inplace=True),
        )
        
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(init_feat_num, 0.8),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(init_feat_num, init_feat_num, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(init_feat_num, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(init_feat_num, init_feat_num//2, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(init_feat_num//2, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(init_feat_num//2, init_feat_num//4, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(init_feat_num//4, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(init_feat_num//4, opt.channels, 3, stride=1, padding=1),
        #     nn.Tanh(),
        # )
        model = [
            nn.BatchNorm2d(init_feat_num, 0.8),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(init_feat_num, init_feat_num//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(init_feat_num//2, 0.8),
            nn.ReLU(inplace=True),
        ]

         # Residual blocks
        for _ in range(2):
            model += [ResBlock(init_feat_num//2)]

        model +=[
            nn.Upsample(scale_factor=2),
            nn.Conv2d(init_feat_num//2, init_feat_num//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(init_feat_num//4, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_feat_num//4, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        ]

        self.conv = nn.Sequential(*model)
    def forward(self, z):
        z = self.linear(z)
        z = z.view(z.shape[0], init_feat_num, self.init_size, self.init_size)
        img = self.conv(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def discriminator_block(in_feat, out_feat, normalize=True):
            block = [
                nn.Conv2d(in_feat, out_feat, 3, 2, 1),
            ]
            if normalize:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, normalize=False),
            *discriminator_block(16, 32),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = 12

        self.clss = nn.Sequential(
            nn.Linear(128*ds_size**2, 1), nn.Sigmoid()
        )


    def forward(self, img):
        out = self.model(img)
        # flatten
        out = out.view(img.shape[0], -1)
        validity = self.clss(out)
        return validity

class HorseDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Define last batch loss
training_bool = True

# Loss function
adversarial_loss = nn.BCELoss()

# Instantiate two networks
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Training data
os.makedirs("data/horse", exist_ok=True)

horseDataset = HorseDataset('data/horse/labels/labels.csv', 'data/horse/images/gan')

dataloader = torch.utils.data.DataLoader(
    horseDataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

## Training

for epoch in range(opt.n_epochs):
    last_loss = 0.0
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial labels
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Variablize input
        real_imgs = Variable(imgs.type(Tensor))

        # Sample gaussian noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.size(0), opt.g_input_size))))

        # Generate a batch of images
        gen_imgs = generator(z)

        #####################
        ## Train Generator ##
        #####################
        
        if not training_bool:
            optimizer_G.zero_grad()

            # calculate g_loss
            gd_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss = gd_loss

            g_loss.backward()
            optimizer_G.step()

            if (last_loss-g_loss<1.0 and last_loss<1.0 and g_loss<1.0):
                training_bool=not training_bool
            
            last_loss = g_loss

            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item())
            )

        #########################
        ## Train Discriminator ##
        #########################
        if training_bool:

            optimizer_D.zero_grad()
            # Generate a batch of images
            gen_imgs = generator(z)

            # calculate d_loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss)/2

            d_loss.backward()
            optimizer_D.step()

            if (last_loss-d_loss<0.1 and last_loss<0.1 and d_loss<0.1):
                training_bool=not training_bool
            
            last_loss = d_loss

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item())
            )


        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:min(16, int(opt.batch_size/2))], "images/horse/%d.png" % batches_done, nrow=5, normalize=True)
