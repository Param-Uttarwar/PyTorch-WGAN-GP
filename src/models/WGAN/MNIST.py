# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:08:24 2020
WGAN network and training for MNIST dataset
@author: Param
"""
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import *
from src.helper.visualize import *
import numpy as np
import os
import pdb


#Descriminator Network
class Descrim(nn.Module):
    def __init__(self):
        super(Descrim, self).__init__()
        
        self.conv=nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,64, 4, 2, 1)
            )
        
        self.fc=nn.Sequential(
            nn.Linear(64,25),
            nn.LeakyReLU(True),
            nn.Linear(25,1)
            )
               
    def forward(self,data):
        out=self.conv(data)
        out=self.fc(out.squeeze())
        return out


    
#Generator Network

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64)
            )
        
        self.genconv=nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4,2, 4, 2, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(2,1, 4, 2, 1),
            nn.Tanh()
        )

        
    def forward(self,data):
        out=self.fc(data)
        out=out.reshape((-1,64,1,1))
        out=self.genconv(out)
        return out
        
    
class model():
    
    def __init__(self,parameters,trainset=None):
        
        self.parameters=parameters
        

        #Check for cuda
        if torch.cuda.is_available():   
          dev = "cuda:0" 
        else:  
          dev = "cpu"  
        self.device = torch.device(dev)
        
        
        #Generator
        self.modelG=Generator().to(self.device)  
        self.parameters['criterionG']=nn.MSELoss()  
        self.parameters['optimizerG']=torch.optim.Adam(self.modelG.parameters(), \
                                             lr=self.parameters['learning_rateG'], weight_decay=4e-3)
        
        #Descriminator
        self.modelD=Descrim().to(self.device)               
        self.parameters['criterionD']=nn.MSELoss() 
        
        self.parameters['optimizerD']=torch.optim.Adam(self.modelD.parameters(), \
                                             lr=self.parameters['learning_rateD'], weight_decay=4e-3)

        
        #Initiate PyTorch dataloaders
        if trainset:
            self.trainset=trainset
            self.train_loader = torch.utils.data.DataLoader(dataset=self.trainset, 
                                                   batch_size=self.parameters['batch_size'], 
                                                   shuffle=True)
    

    def train(self):
        losses={'G':[],'D':[]}
         #Iterate over epochs
        for epoch in range(self.parameters['num_epochs']):
            tmp_lossG,tmp_lossD=[],[]
            
            #Iterate over batches
            for i, (data,_) in enumerate(self.train_loader):
                
                #Flush Gradients
                self.parameters['optimizerD'].zero_grad()
                
                #Convert data
                data = data.to(self.device)
                
                
                #UPDATE DESCRIMINATOR
                for j in range(5):
                    #For real 
                    inpD=data
                    labelsD=torch.ones(data.shape[0]).type(torch.FloatTensor).to(self.device)  
                    
                    #Discriminator pred
                    predD=self.modelD(inpD).squeeze()
                    
                    #Calculate loss                
                    lossDR = self.parameters['criterionD'](predD, labelsD)
                    lossDR.backward()    
    
                    #For Fake
         
                    #Generate fake data
                    noise=torch.rand(data.shape[0],64).type(torch.FloatTensor).to(self.device)
                    predG=self.modelG(noise)
                    
                    inpD=predG
                    labelsD=torch.zeros(data.shape[0],1).type(torch.FloatTensor).to(self.device)  
                    
                    #Discriminator pred
                    predD=self.modelD(inpD.detach())
                    
                    #Calculate loss                
                    lossDF = self.parameters['criterionD'](predD, labelsD)
                    lossDF.backward() 
                    
                    #Update gradients
                    self.parameters['optimizerD'].step()
                    tmp_lossD.append(lossDR.item()+lossDF.item())
    
                    #Weight Clipping
                    for p in self.modelD.parameters():
                        p.data.clamp_(-0.1, 0.1)                
                
                
                #UPDATE GENERATOR
                    
                #Flush gradients
                self.parameters['optimizerG'].zero_grad()
                
                
                #Make data
                inpG=self.modelG(noise)
                labelsG=torch.ones((data.shape[0],1))
                labelsG = labelsG.type(torch.FloatTensor).to(self.device)
                
                #D(G(z)) pred
                predDG=self.modelD(inpG)
                               
                #Calculate loss                
                loss = self.parameters['criterionD'](predDG, labelsG)
                
                #Update Gradients
                loss.backward()
                self.parameters['optimizerG'].step()
                tmp_lossG.append(loss.item())
            
            #Checkpoint save    
            if epoch%(self.parameters['num_epochs']//4)==0:
                torch.save(self.modelG.state_dict(), os.path.join(os.getcwd(),'results','models',self.parameters['model'],'%s_checkpoint_%d.model'%(self.parameters['dataset'],epoch)))
            


            #Temporary Visualization                
            if epoch%100==0:
                pass
                #Do visualization
               
            
            #Store loss after every epoch
            losses['G'].append(np.mean(tmp_lossG))
            losses['D'].append(np.mean(tmp_lossD))
            print("Epoch : %d Loss Generator : %f Loss Discrim : %f \
                  "%(epoch,np.mean(tmp_lossG),np.mean(tmp_lossD)))
        
        #Save losses and model
        self.losses=losses
        torch.save(self.modelG.state_dict(), os.path.join(os.getcwd(),'results','models',self.parameters['model'],'%s.model'%self.parameters['dataset']))
        losses_df=pd.DataFrame(losses)
        losses_df.to_excel(os.path.join(os.getcwd(),'results','models',self.parameters['model'],'%s_losses.xlsx'%self.parameters['dataset']))    
        
        
        
        