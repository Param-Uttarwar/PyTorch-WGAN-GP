# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:06:15 2020
Functions to visualize and save images
@author: Param
"""

import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import torch

def visualize(model):
    """
    

    Parameters
    ----------
    model : Network model class
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    size=5  #show 5*5=25 images in the figure
    
    #Make some noise
    noise=torch.randn(size**2,64)
    
    #Generate images
    images=model.modelG(noise)
    images=images.detach().numpy()
    image_size=images.shape[-1]
    images=images.reshape((size,size,image_size,image_size))
    
    #Create tiled image
    tmp_imgs=[]
    for i in range(5):
        tmp_imgs.append(np.vstack(tuple([images[i,j,:,:] for j in range(size)])))
    all_image=np.hstack(tuple(tmp_imgs))
    
    plt.imshow(all_image) 
    
    
