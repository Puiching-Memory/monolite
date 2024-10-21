import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.abspath("./"))
from lib.utils.metrics import bbox_iou


def decode_bbox(
    boxes_tensor: torch.Tensor, device: str = "cpu", stride: int = 8
) -> torch.Tensor:
    """
    将模型box2d头输出解码,ltrb
    ---
    input:
        boxes_tensor: 模型输出的box2d头,shape:(B, 4*reg_max, H, W)
    output:
        解码后的ltrb,shape:(B, H, W, 4)
    doc:
        https://zhuanlan.zhihu.com/p/633094573
        YOLOv8,等距积分
    """
    B, reg_max, H, W = boxes_tensor.shape
    reg_max = reg_max // 4

    boxes_tensor = boxes_tensor.reshape(B, H, W, 4, reg_max)
    scale = torch.linspace(0, reg_max - 1, reg_max, device=device)  # (reg_max) 0~15

    scale = scale.repeat(B, 1, H, W).reshape(B, H, W, 1, reg_max)
    boxes_tensor = scale * boxes_tensor  # (B,H,W,4,reg_max)
    boxes_tensor = torch.sum(boxes_tensor, dim=4)  # (B,H,W,4)
    boxes_tensor = boxes_tensor * stride  # 放大到原图尺度

    return boxes_tensor


def notassigner(
    pred_box2d: torch.Tensor, pred_offset: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    正负样本匹配
    ---
    多预测框->单真实框

    target:
        GT,shape:(B,num_gt,6) (x1,y1,x2,y2,x,y)
    pred_box2d:
        模型输出的box2d头,shape:(B, 4, H, W)
    pred_offset:
        模型输出的offset头,shape:(B, 2, H, W)

    1. 依据box2d和offset计算出预测框,shape:(B,H,W,6)
    2. 计算与GT的距离,shape:(B,H*W,num_gt)
    3. 挑选出距离最近的GT,得到索引矩阵,shape:(B,H*W)
    4. 依据索引矩阵,生成GT矩阵,shape:(B,6,H,W)

    return:
        匹配后的GT矩阵,shape:(B,6,H,W)
    """
    B, _, H, W = pred_box2d.shape
    num_gt = target.shape[1]

    pred_box2d = torch.cat([pred_box2d, pred_offset], dim=1)  # (B,4+2,H,W)
    pred_box2d = pred_box2d.reshape(B, H * W, 6)  # (B,H*W,6)

    distance = torch.cdist(pred_box2d[-2:], target[-2:]) # (B,H*W,num_gt)
    index = torch.argmin(distance, dim=2)  # (B,H*W)
    
    gt = torch.gather(target, 1, index.unsqueeze(2).repeat(1, 1, 6))  # (B,H*W,6)
    gt = gt.reshape(B, 6, H, W)  # (B,6,H,W)

    return gt


if __name__ == "__main__":
    pred_box2d = torch.rand(2, 4, 48, 160)
    pred_offset = torch.rand(2, 2, 48, 160)
    target = torch.rand(2, 2 , 6)
    result = notassigner(pred_box2d, pred_offset, target)
    print(result.shape)
