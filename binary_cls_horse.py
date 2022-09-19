import os
import sys
from statistics import mean
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

epochs = 200
batch_size = 32
lr = 0.001
b1 = 0.5
b2 = 0.99
img_size = 224
channels = 3
resume = True

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2,2)),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2,2)),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((8,8)),

        )
        self.linear = nn.Sequential(
            nn.Linear(12544, 1000),
            nn.BatchNorm1d(1000, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.conv(img)
        out = out.view(img.shape[0], -1)
        out = self.linear(out)
        return out
    

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

# define loss function
bc_loss = nn.BCELoss()
# instantiate the network
horse_classifier = CNN()

if cuda:
    horse_classifier.cuda()

# Training data
os.makedirs("data/horse", exist_ok=True)

horseDataset = HorseDataset('data/horse/labels/labels.csv', 'data/horse/images/bc')

dataloader = DataLoader(
    horseDataset,
    batch_size=batch_size,
    shuffle=True,
)

# Optimizers
optimizer = torch.optim.Adam(horse_classifier.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

losses = []
accur = []

# Initialize weights
if resume:
    horse_classifier.load_state_dict(torch.load('data/horse/model/model_horse_cls.pt'))
    horse_classifier.eval()
else:
    horse_classifier.apply(weights_init_normal)

img = read_image('images/horse/400.png')
img = img.view(1, 3, 224, 224)
print(img)

out = horse_classifier(img.type(Tensor))
print(out)

# # Training
# for epoch in range(epochs):
#     for i, (imgs, labels) in enumerate(dataloader):
#         # variablize input
#         input = Variable(imgs.type(Tensor))
#         # calculate output
#         out = horse_classifier(input)
#         # loss
#         loss = bc_loss(out, labels.type(Tensor).reshape(-1,1))

#         # accuracy
#         acc = np.mean(np.squeeze(Tensor.cpu(out).detach().numpy().round()==(labels.numpy().reshape(-1,1))).astype(int))

#         # backprop
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # statistics
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [loss: %f] "
#             % (epoch, epochs, i, len(dataloader), loss.item())
#         )
        
#     if (epoch%10==0):
#         losses.append(loss)
#         accur.append(acc)
#         print("epoch {}\tloss : {}\t accuracy : {}".format(epoch,loss,acc))
#         torch.save(horse_classifier.state_dict(), 'data/horse/model/model_horse_cls.pt')
