import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.utils.assigner import notassigner
from lib.utils.metrics import bbox_iou
from lib.models.loss import VFLoss,FocalLoss,Poly1FocalLoss,focal_loss_cornernet
from focal_loss.focal_loss import FocalLoss

class loss(object):
    def __init__(self,device):
        self.device = device
        #self.heatmap_loss = nn.BCEWithLogitsLoss()
        self.cls2d_loss = nn.BCEWithLogitsLoss()
    
    def loss(self,output:dict[torch.Tensor], target:dict[torch.Tensor])->torch.Tensor:    
        """
        计算损失函数
        ---
        output:dict[torch.Tensor]
            keys:
            - 'box2d': 预测的2D框坐标,shape为(batch_size,num_anchors,4)
            - 'offset2d': 预测的2D框中心偏移量,shape为(batch_size,num_anchors,2)
            - 'cls2d': 预测的2D框类别,shape为(batch_size,num_anchors,num_classes)
        
        return:torch.Tensor
            损失函数值
        """
        #loss_box2d = -torch.log(bbox_iou(output['box2d'], target['box2d']))
        #loss_temp = nn.BCEWithLogitsLoss()(output['neck'],torch.zeros_like(output['neck']))
        loss_heatmap = focal_loss_cornernet(output['heatmap'].sigmoid_(), target['heatmap'])
        
        loss = loss_heatmap
        loss_info = {'loss_heatmap':loss_heatmap}
           
        return loss,loss_info