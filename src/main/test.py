# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:00:19 2020
testing script 
@author: Param

"""


import numpy as np 
import os
import importlib.util
import torch
import torch.nn as nn
from src.helper.visualize import *


if __name__ == "__main__":

#------------------------SET PARAMETERS---------------------------------------

    dataset="MNIST" #dataset =["MNIST","CIFAR10","FashionMNIST"]
    model='GAN' #model=['GAN','WGAN','WGAN-GP']
              
    parameters={'learning_rateG':5e-4, 
            'learning_rateD':5e-4,
            'num_epochs':20,
            'batch_size':64,
            'dataset':dataset, 
            'model':model 
            }  
    
#-----------------------------------------------------------------------------

#----------------------LOAD MODEL----------------------------------------------    
   
    spec = importlib.util.spec_from_file_location("model", os.path.join(os.getcwd(),'src','models',model,'%s.py'%(dataset)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_test=module.model(parameters)

#-----------------------------------------------------------------------------

    visualize(model_test)
