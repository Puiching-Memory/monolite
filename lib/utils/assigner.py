import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath("./"))
    
def decode_bbox(boxes_tensor:torch.Tensor,device:str='cpu',stride:int=8)->torch.Tensor:
    """
    将模型box2d头输出解码
    ---
    input:
        boxes_tensor: 模型输出的box2d头,shape:(B, 4*reg_max, H, W)
    output: 
        解码后的distance(ltrb),shape:(B, H, W, 4)
    doc:
        https://zhuanlan.zhihu.com/p/633094573
        YOLOv8,等距积分
    """
    B,reg_max,H,W = boxes_tensor.shape
    reg_max = reg_max//4
    
    boxes_tensor = boxes_tensor.reshape(B,H,W,4,reg_max)
    scale = torch.linspace(0,reg_max-1,reg_max,device=device) # (reg_max)
    scale = scale.repeat(B,1,H,W).reshape(B,H,W,1,reg_max)
    boxes_tensor = scale * boxes_tensor # (B,H,W,4,reg_max)
    boxes_tensor = torch.sum(boxes_tensor,dim=4) # (B,H,W,4)
    boxes_tensor = boxes_tensor * stride # 放大到原图尺度
    
    return boxes_tensor


def bbox_decode(anchor_points, pred_dist):
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    proj = torch.arange(8, dtype=torch.float)
    b, a, c = pred_dist.shape  # batch, anchors, channels
    pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    return dist2bbox(pred_dist, anchor_points, xywh=False)

if __name__ == '__main__':
    from lib.utils.metrics import bbox_iou,dist2bbox,make_anchors
    reg_max = 8
    raw_feature = torch.rand(2, 4*reg_max, 48, 160)
    pred_dist = raw_feature.reshape(2, 48*160, 4*reg_max)
    
    anchor_points, stride_tensor = make_anchors([raw_feature],strides=[8])

    boxes_tensor = bbox_decode(anchor_points,pred_dist)
    print(boxes_tensor)
    
    boxes_tensor = dist2bbox(boxes_tensor,anchor_points)

        
    
    