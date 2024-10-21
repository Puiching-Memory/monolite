import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os

sys.path.append(os.path.abspath("./"))

class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)

def focal_loss_cornernet(input, target, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    '''

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    loss = 0

    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * neg_weights

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()
    
class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(VFLoss, self).__init__()
        # 传递 nn.BCEWithLogitsLoss() 损失函数  must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn  #
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply VFL to each element

    def forward(self, pred, true):

        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits

        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (
                    true <= 0.0).float()
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

if __name__ == "__main__":
    pass
