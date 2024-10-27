import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
import cv2


class visualizer(object):
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

    def decode(
        self, image: torch.Tensor, output: dict[torch.Tensor]
    ) -> dict[np.ndarray]:
        # 解码image
        image = self.image_transforms(image)
        image = image[0].permute(1, 2, 0).cpu().detach().numpy()  # (384, 1280, 3)

        # 解码heatmap
        heatmap = output["heatmap"][0].sigmoid_().cpu().detach().numpy()  # (C, 48, 160)
        heatmap = heatmap * 255.0
        heatmap = heatmap.astype(np.uint8)
        print(heatmap)

        # 拆分heatmap
        heatmap_backgroud = heatmap[0]
        heatmap_cls1 = heatmap[1]

        # 将heatmap缩放至原图大小
        heatmap_backgroud = cv2.resize(heatmap_backgroud, (1280, 384))
        heatmap_cls1 = cv2.resize(heatmap_cls1, (1280, 384))

        # 映射颜色图
        heatmap_backgroud = cv2.applyColorMap(
            heatmap_backgroud, cv2.COLORMAP_VIRIDIS
        )
        heatmap_cls1 = cv2.applyColorMap(
            heatmap_cls1, cv2.COLORMAP_VIRIDIS
        )

        # 合并图像
        image_backgroud = cv2.addWeighted(image, 0.5, heatmap_backgroud, 0.5, 0)
        image_cls1 = cv2.addWeighted(image, 0.5, heatmap_cls1, 0.5, 0)
        # print(heatmap, heatmap.shape)

        cv2.imwrite(os.path.join(self.save_path, "temp_backgroud.jpg"), image_backgroud)
        cv2.imwrite(os.path.join(self.save_path, "temp_cls1.jpg"), image_cls1)

        return {"image_backgroud": image_backgroud, "image_cls1": image_cls1}
