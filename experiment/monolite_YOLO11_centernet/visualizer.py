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

    def decode(self, image: torch.Tensor, output: dict[torch.Tensor]):
        image = self.image_transforms(image)
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()  # (384, 1280, 3)

        heatmap = output["heatmap"].sigmoid_().squeeze().cpu().detach().numpy()  # (48,160)
        heatmap = heatmap * 255.0
        heatmap = cv2.resize(heatmap, (1280, 384))
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        print(heatmap, heatmap.shape)

        cv2.imwrite(os.path.join(self.save_path, "temp.jpg"), image)
        
        return image