import sys
import os

sys.path.append(os.path.abspath("./"))
from lib.cfg.base import VisualizerBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
import cv2


class visualizer(VisualizerBase):
    def __init__(self):
        self.save_path = (
            r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet\output"
        )

        self.image_transforms = v2.Compose(
            [
                v2.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(384, 1280)),
            ]
        )

    def decode_output(
        self, image: torch.Tensor, output: tuple[torch.Tensor, ...]
    ) -> dict[np.ndarray]:
        # 解码image
        image = self.image_transforms(image)
        image = image[0].permute(1, 2, 0).cpu().detach().numpy()  # (384, 1280, 3)

        # 解码heatmap
        heatmap = output[6][0].sigmoid_().cpu().detach().numpy()  # (C, 48, 160)
        heatmap = heatmap * 255.0
        heatmap = heatmap.astype(np.uint8)

        # 拆分heatmap
        heatmap_backgroud = heatmap[0]
        heatmap_cls1 = heatmap[1]

        # 将heatmap缩放至原图大小
        heatmap_backgroud = cv2.resize(heatmap_backgroud, (1280, 384))
        heatmap_cls1 = cv2.resize(heatmap_cls1, (1280, 384))

        # 映射颜色图
        heatmap_backgroud = cv2.applyColorMap(heatmap_backgroud, cv2.COLORMAP_VIRIDIS)
        heatmap_cls1 = cv2.applyColorMap(heatmap_cls1, cv2.COLORMAP_VIRIDIS)

        # 合并图像
        image_backgroud = cv2.addWeighted(image, 0.5, heatmap_backgroud, 0.5, 0)
        image_cls1 = cv2.addWeighted(image, 0.5, heatmap_cls1, 0.5, 0)
        # print(heatmap, heatmap.shape)

        cv2.imwrite(
            os.path.join(self.save_path, "output_backgroud.jpg"), image_backgroud
        )
        cv2.imwrite(os.path.join(self.save_path, "output_cls1.jpg"), image_cls1)

        return {"output_backgroud": image_backgroud, "output_cls1": image_cls1}

    def decode_target(
        self, image: torch.Tensor, output: dict[torch.Tensor]
    ) -> dict[np.ndarray]:
        # 解码image
        image = self.image_transforms(image)
        image = image[0].permute(1, 2, 0).cpu().detach().numpy()  # (384, 1280, 3)

        # 解码heatmap
        heatmap = output["heatmap"][0].sigmoid_().cpu().detach().numpy()  # (C, 48, 160)
        heatmap = heatmap * 255.0
        heatmap = heatmap.astype(np.uint8)

        # 拆分heatmap
        heatmap_backgroud = heatmap[0]
        heatmap_cls1 = heatmap[1]

        # 将heatmap缩放至原图大小
        heatmap_backgroud = cv2.resize(heatmap_backgroud, (1280, 384))
        heatmap_cls1 = cv2.resize(heatmap_cls1, (1280, 384))

        # 映射颜色图
        heatmap_backgroud = cv2.applyColorMap(heatmap_backgroud, cv2.COLORMAP_VIRIDIS)
        heatmap_cls1 = cv2.applyColorMap(heatmap_cls1, cv2.COLORMAP_VIRIDIS)

        # 合并图像
        image_backgroud = cv2.addWeighted(image, 0.5, heatmap_backgroud, 0.5, 0)
        image_cls1 = cv2.addWeighted(image, 0.5, heatmap_cls1, 0.5, 0)
        # print(heatmap, heatmap.shape)

        cv2.imwrite(
            os.path.join(self.save_path, "target_backgroud.jpg"), image_backgroud
        )
        cv2.imwrite(os.path.join(self.save_path, "target_cls1.jpg"), image_cls1)

        return {"target_backgroud": image_backgroud, "target_cls1": image_cls1}
