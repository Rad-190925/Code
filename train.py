%reload_ext autoreload
%autoreload 2
%matplotlib inline

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import numpy as np
import fastai
from fastai import vision, data  
from fastai.data import DataBunch
from fastai.vision import *
from fastai.data import *
from fastai.metrics import *
from fastai.train import *

model = tvm.densenet161

learner = ConvLearner(hipOA_labelbunch, model, metrics=[accuracy], pretrained = True, ps = 0.4, callback_fns=[ShowGraph, BnFreeze])

class MTL_block (nn.Module):
    def __init__(self, in_features = 1000):
        super (MTL_block, self).__init__()
        self.FOS = nn.Linear(in_features, 4)
        self.AOS = nn.Linear(in_features, 4)
        self.JSN = nn.Linear(in_features, 4)
        self.SUBSCL = nn.Linear(in_features, 2)
        self.SUBCYST = nn.Linear(in_features, 2)
        
    def forward(self, x):
        FOS = self.FOS(x)
        AOS = self.AOS (x)
        JSN = self.JSN(x)
        SUBSCL = self.SUBSCL (x)
        SUBCYST = self.SUBCYST (x)
        
        return FOS, AOS, JSN, SUBSCL, SUBCYST
        
learner.model[1][8] = MTL_block()

def multitask_loss(input, target):
    input_FOS = input [0]
    input_AOS = input [1]
    input_JSN = input [2]
    input_SUBSCL = input [3]
    input_SUBCYST = input [4]
    
    target_FOS = target[:,0]
    target_AOS = target[:,1]
    target_JSN = target[:,2]
    target_SUBSCL = target[:,3]
    target_SUBCYST = target[:,4]
    cross_entropy = nn.CrossEntropyLoss()
    loss_FOS = cross_entropy (input_FOS, target_FOS.long())
    loss_AOS = cross_entropy (input_AOS, target_AOS.long())
    loss_JSN = cross_entropy (input_JSN, target_JSN.long())
    loss_SUBSCL = cross_entropy (input_SUBSCL, target_SUBSCL.long())
    loss_SUBCYST = cross_entropy (input_SUBCYST, target_SUBCYST.long())    
    
    return loss_FOS/4 + loss_AOS/4 + loss_JSN/4 + loss_SUBSCL/16 + loss_SUBCYST/16     
    
learner.loss_fn = multitask_loss

lrf = 1e-3
epochs = 50

learner.fit_one_cycle (epochs, lrf)
