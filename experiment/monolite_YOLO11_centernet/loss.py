import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.cfg.base import LossBase
from lib.utils.assigner import notassigner
from lib.utils.metrics import bbox_iou
from lib.models.loss import (
    Poly1FocalLoss,
    VFLoss,
    focal_loss_cornernet,
    MultiFocalLoss,
    BinaryFocalLoss,
)
from lib.utils.logger import logger


class loss(LossBase):
    def __init__(self):
        # self.heatmap_loss = VFLoss(nn.BCEWithLogitsLoss())
        self.heatmap_loss = focal_loss_cornernet
        self.cls2d_loss = nn.BCEWithLogitsLoss()

    def __call__(
        self, output: dict[torch.Tensor], target: dict[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[torch.Tensor]]:
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
        # loss_box2d = -torch.log(bbox_iou(output['box2d'], target['box2d']))
        # loss_temp = nn.BCEWithLogitsLoss()(output['neck'],torch.zeros_like(output['neck']))

        # loss_heatmap = focal_loss_cornernet(
        #     torch.clamp(output["heatmap"].sigmoid_(), min=1e-4, max=1 - 1e-4),
        #     target["heatmap"],
        # )
        loss_heatmap = self.heatmap_loss(
            torch.sigmoid(output["heatmap"].float()),
            target["heatmap"].float(),
        )

        loss = loss_heatmap
        loss_info = {"loss_heatmap": loss_heatmap}

        logger.info(output["heatmap"])
        logger.warning(target["heatmap"])
        print(output["heatmap"].shape, target["heatmap"].shape)
        logger.info(loss)

        return loss, loss_info

    def __str__(self):
        return f"{self.__class__.__name__}\n{vars(self)}"


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = loss()
    print(loss_fn)
