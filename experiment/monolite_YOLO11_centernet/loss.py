import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.utils.assigner import decode_bbox
from lib.models.loss import BboxLoss

class loss(object):
    def __init__(self,device,stride):
        from lib.utils.assigner import TaskAlignedAssigner
        self.device = device
        self.stride = stride
        self.assigner = TaskAlignedAssigner(topk=10,num_classes=3,alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max=8).to(device)
        self.cls2d_loss = nn.BCEWithLogitsLoss()
    
    def loss(self,output:dict[torch.Tensor], target:dict[torch.Tensor])->torch.Tensor:        
        output['box2d'] = decode_bbox(output['box2d'],self.device,self.stride)
        
        cls2d_loss = self.cls2d_loss(output['cls2d'], torch.zeros_like(output["cls2d"]))
        
        loss1 = nn.L1Loss()(output["neck"], torch.zeros_like(output["neck"]))
        
        loss_info = {'loss1':loss1.item(),'cls2d_loss':cls2d_loss.item()}
        return loss1+cls2d_loss,loss_info