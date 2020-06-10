# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:26:36 2020

Script and functions to create,download and access datasets

Options : MNIST, CIFAR-10, Toy
@author: Param

"""
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

def choose_dataset(option,download=False):
    """
    

    Parameters
    ----------
    option : string
        Choose from ["MNIST","CIFAR10","FashionMNIST"].
    download : boolean, optional
        False if already downloaded the dataset in /data. The default is False.

    Returns
    -------
    trainset : train dataset
        PyTorch dataset.
    valset : validation set
        PyTorch dataset.

    """
    assert option in ["MNIST","CIFAR10","FashionMNIST"]
    
    if option=="MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST(os.path.join(os.getcwd(),'data'), \
                                  download=download, train=True,transform=transform)
        
    elif option=="CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.CIFAR10(os.path.join(os.getcwd(),'data')\
                                    , download=download, train=True,transform=transform)
        
    elif option=="FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.FashionMNIST(os.path.join(os.getcwd(),'data'),\
                                         download=download, train=True,transform=transform)
        
    
    return trainset