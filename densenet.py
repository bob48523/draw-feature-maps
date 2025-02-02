import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

FLOPS = 0

class se(nn.Module):
    def __init__(self, in_planes):
        super(se,self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes)
        self.fc2 = nn.Linear(in_planes, in_planes)
    
    def forward(self,x):
        global FLOPS
        out = F.avg_pool2d(x, kernel_size=x.size(2))
        out = self.fc1(out.view(out.size(0),-1))
        FLOPS += self.fc1.in_features*self.fc1.out_features
        out = self.fc2(out)
        FLOPS += self.fc2.in_features*self.fc2.out_features
        out = F.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out.repeat(1,1,x.size(2),x.size(3))
        #print(out)
        return out*x

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.shuffle1 = ShuffleBlock(groups=6)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1,groups= 6, bias=False)
        self.bn3 = nn.BatchNorm2d(growthRate)
        self.se = se(growthRate)

    def forward(self, x):
        global FLOPS
        out = self.bn1(x)
        out = self.conv1(out)
        FLOPS += self.conv1.in_channels*self.conv1.out_channels*self.conv1.kernel_size[0]*self.conv1.kernel_size[1]*out.size(2)*out.size(3)
        #out = self.shuffle1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        FLOPS += self.conv2.in_channels*self.conv2.out_channels*self.conv2.kernel_size[0]*self.conv2.kernel_size[1]*out.size(2)*out.size(3)//6
        out = self.bn3(out)
        out = self.se(out)
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        global FLOPS
        out = self.conv1(F.relu(self.bn1(x)))
        FLOPS += self.conv1.in_channels*self.conv1.out_channels*self.conv1.kernel_size[0]*self.conv1.kernel_size[1]*out.size(2)*out.size(3)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        global FLOPS
        FLOPS = 0
        out = self.conv1(x)
        FLOPS += self.conv1.in_channels*self.conv1.out_channels*self.conv1.kernel_size[0]*self.conv1.kernel_size[1]*out.size(2)*out.size(3)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        FLOPS += self.fc.in_features*self.fc.out_features
        return out

def CalcuFlops():
    net = DenseNet(growthRate=24, depth=88, reduction=0.5, bottleneck=True, nClasses=10)
    x = torch.randn(3,3,32,32)
    y = net(Variable(x))
    print ('flops:%d'%FLOPS)
