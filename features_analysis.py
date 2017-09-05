#!/usr/bin/env python3

import torch

import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import sys
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import densenet

def LoadData():
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} 
    trainLoader = DataLoader(
        dset.CIFAR100(root='cifar', train=True, download=True,
                     transform=trainTransform),
        batch_size=64, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dset.CIFAR100(root='cifar', train=False, download=True,
                     transform=testTransform),
        batch_size=64, shuffle=False, **kwargs)
    return trainLoader, testLoader

trainLoader, testLoader = LoadData() 
net = torch.load('work/dense_se_net2.base/latest.pth')
for data, target in testLoader:
    break

data, target = data.cuda(), target.cuda()
data, target = Variable(data, volatile=True), Variable(target)

conv1_layer = net.module._modules.get('conv1')
print(conv1_layer)
features = torch.zeros(64,48,32,32)

def fun(m, i, o): features.copy_(o.data)

conv1_hook = conv1_layer.register_forward_hook(fun)
h_x = net(data)

plt.figure(figsize=(10, 10), facecolor='w')
plt.title('featuremap')
plt.imshow(np.asarray(features.numpy()[0,1,:,:]))
#plt.imshow(np.asarray(features.numpy()[0,1,:,:]),cmap=mpl.cm.gray)
plt.savefig('features3.png')

#conv1_hook.remove()
