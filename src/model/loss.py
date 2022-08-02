import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# Contrast loss is ordinary cross entropy, so it can be done in place. Unlike these. 

def masked_orientation_loss_fn(pred,target,weight,unlabeled):
    """
    For the loss of the orientation, use everything except the unlabeled/oblique class. 
    """

    target_ = target[target.argmax(dim=1) != unlabeled]
    pred = pred[target.argmax(dim=1) != unlabeled]
    return F.cross_entropy(pred, torch.max(target_, 1)[1],weight=weight)

def contrast_loss_fn(pred,target):
    return F.cross_entropy(pred, target)