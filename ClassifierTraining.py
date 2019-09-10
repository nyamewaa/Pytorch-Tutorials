#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:58:42 2019
Training a classifier using CIFAR10 data set
@author: nyamewaa
"""
#%% IMPORTING REQUIRED PACKAGES
import torch
import torchvision
import torchvision.transforms as transforms

#%% LOADING ICIFAR10 DATA AND SETTING CLASS NAMES

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=2)
classes=('plane','car','bird','cat',
         'deer','dog','frog','horse','ship','truck')
 #%% 
