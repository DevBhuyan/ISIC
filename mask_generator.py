#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:09:06 2023

@author: dev
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.init
import os

use_cuda = torch.cuda.is_available()


base_dir = '/mnt/C654078D54078003/Curious Dev B/PROJECT STAGE - II/ISIC 2016 for maskgen/'


def clahe(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return clahe_image


def dullrazor(clahe_image):
    img = clahe_image
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(9,9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)  
    return dst 


def generator(dst):
    nChannel = 10
    maxIter = 800
    minLabels = 1
    lr = 0.1
    nConv = 2
    visualize = 1
    stepsize_sim = 1
    stepsize_con = 1
    
    # CNN model
    class MyNet(nn.Module):
        def __init__(self,input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
            self.bn1 = nn.BatchNorm2d(nChannel)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(nConv-1):
                self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
                self.bn2.append( nn.BatchNorm2d(nChannel) )
            self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
            self.bn3 = nn.BatchNorm2d(nChannel)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu( x )
            x = self.bn1(x)
            for i in range(nConv-1):
                x = self.conv2[i](x)
                x = F.relu( x )
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x
    
    im = dst
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    
    
    # train
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
    model.train()
    
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average = True)
    loss_hpz = torch.nn.L1Loss(size_average = True)
    
    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, nChannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.array([
                            [0, 0, 0],
                            [255, 255, 255],
                            [255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [128, 128, 128],
                            [192, 192, 192]
        ])
    
    for batch_idx in range(maxIter):
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    
        outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)
    
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            cv2.imshow( "output", im_target_rgb )
            cv2.waitKey(10)
    
        # loss 
        loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
            
        loss.backward()
        optimizer.step()
    
        print (batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())
    
        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            im_target_rgb = buffer
            break
        
        buffer = im_target_rgb
    
    # save output image
    if not visualize:
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )

    return im_target_rgb

for file in os.listdir(base_dir+'IMAGES/'):
    clahe_image = clahe(base_dir+'IMAGES/'+file)
    dst = dullrazor(clahe_image)
    im_target_rgb = generator(dst)
    cv2.imwrite(base_dir+'MASKS/'+file, im_target_rgb)
    break
