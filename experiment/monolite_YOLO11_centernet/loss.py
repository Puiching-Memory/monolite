import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.utils.assigner import decode_bbox

class loss(object):
    def __init__(self,device,stride):
        from lib.utils.assigner import TaskAlignedAssigner
        self.device = device
        self.stride = stride
        self.assigner = TaskAlignedAssigner(topk=10,num_classes=3)
        self.cls2d_loss = nn.BCEWithLogitsLoss()
    
    def loss(self,output:dict[torch.Tensor], target:dict[torch.Tensor])->torch.Tensor:
        #self.assigner(output, target)
        output['box2d'] = output['box2d'].sigmoid()
        output['box2d'] = decode_bbox(output['box2d'],self.device,self.stride)
        loss1 = nn.L1Loss()(output["neck"], torch.zeros_like(output["neck"]))
        return loss1