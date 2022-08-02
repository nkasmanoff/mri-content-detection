import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic Convolution -> Batch Normalization -> ReLU block.
    """
    def __init__(self,inplane,outplane,kernel_size = 7,stride = 1,padding = 1):
        super(BasicBlock, self).__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(inplane,outplane,padding=padding,stride=stride,kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class BrainClassifier(nn.Module):
    """    
    A deep convolutional network capable of labeling brain MRI slices. 

    """
    
    def _make_layer(self,block,inplanes,outplanes,nblocks,kernel_size,stride,padding): 
        layers = []
        for i in range(0,nblocks):
            layers.append(block(inplanes,outplanes,kernel_size,stride,padding))
            inplanes = outplanes 
        return nn.Sequential(*layers)
    
    def __init__(self, block,in_ch=1,ch=16,nblocks = 2,n_contrasts=22,n_orientations = 4):
        super(BrainClassifier,self).__init__()
        self.ch = ch
        #Base / Backbone
        self.layer1 = self._make_layer(block, in_ch, ch, nblocks=nblocks,stride=1,kernel_size=11,padding=0) 
        self.downsample1 = self._make_layer(block, ch, 2*ch, nblocks=1,stride=2,kernel_size=3,padding=0) 
        self.layer2 = self._make_layer(block, 2*ch, 2*ch, nblocks=nblocks,stride=1,kernel_size=9,padding=0)
        self.downsample2 = self._make_layer(block, 2*ch, 2*2*ch, nblocks=1,stride=2,kernel_size=3,padding=0) 
        self.layer3 = self._make_layer(block, 2*2*ch, 2*2*ch, nblocks=1,stride=1,kernel_size=7,padding=0)
        self.downsample3  = self._make_layer(block, 2*2*ch, 2*2*2*ch, nblocks=1,stride=2,kernel_size=3,padding=0)
        self.layer4 = self._make_layer(block, 2*2*2*ch, 2*2*2*ch, nblocks=1,stride=1,kernel_size=5,padding=0)
        self.downsample4  = self._make_layer(block, 2*2*2*ch, 2*2*2*2*ch, nblocks=1,stride=2,kernel_size=3,padding=0)

        # Contrast Classification Head
        self.contrast1 = self._make_layer(block, 2*2*2*2*ch, 2*2*2*2*ch, nblocks=3
                                        ,stride=1,kernel_size=3,padding=0)
        self.contrast_fc1 = nn.Linear(2*2*2*2*ch*2*2, 256) 
        self.contrast_fc2 = nn.Linear(256, n_contrasts) 

        # Orientation Classification Head
        self.orientation1 = self._make_layer(block, 2*2*2*2*ch, 2*2*2*2*ch, nblocks=3
                                        ,stride=1,kernel_size=3,padding=0)
        self.orientation_fc1 = nn.Linear(2*2*2*2*ch*2*2, 256) 
        self.orientation_fc2 = nn.Linear(256, n_orientations) 
       
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.downsample3(x)
        x = self.layer4(x)
        x = self.downsample4(x)
      
        x_contrast = self.contrast1(x)
        x_contrast = x_contrast.view(-1,2*2*2*2*self.ch*2*2) 
        x_contrast = F.relu(self.contrast_fc1(x_contrast))
        x_contrast = self.contrast_fc2(x_contrast)
                
        
        x_orientation = self.orientation1(x)
        x_orientation = x_orientation.view(-1,2*2*2*2*self.ch*2*2) 
        x_orientation = F.relu(self.orientation_fc1(x_orientation))
        x_orientation = self.orientation_fc2(x_orientation)
        

        return x_contrast,x_orientation
    
    

