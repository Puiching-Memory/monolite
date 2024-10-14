import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super().__init__()
        from lib.models import block

        self.backboneP3 = nn.Sequential(
            block.Conv(3, 64, 3, 2),  # 0-P1/2
            block.Conv(64, 128, 3, 2),  # 1-P2/4
            block.C3k2(128, 256, 2, False, 0.25),  # 2
            block.Conv(256, 256, 3, 2),  # 3-P3/8
        )
        self.backboneP4 = nn.Sequential(
            block.C3k2(256, 512, 2, False, 0.25),  # 4
            block.Conv(512, 512, 3, 2),  # 5-P4/16
        )
        self.backboneP5 = nn.Sequential(
            block.C3k2(512, 512, 2, True, 0.5),  # 6
            block.Conv(512, 1024, 3, 2),  # 7-P5/32
        )
        self.backboneTop = nn.Sequential(
            block.C3k2(1024, 1024, 2, True, 0.5),  # 8
            block.SPPF(1024, 1024, 5),  # 9
        )
        self.cls2d = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, 1, 0),
        )
        self.box2d = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4*8, 1, 1, 0),
        )
        self.offset3d = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1, 1, 0),
        )
        self.size3d = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, 1, 0),
        )
        self.heading = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 24, 1, 1, 0),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1, 1, 0),
        )
        self.upsample1 = block.DySample(1024, 2, "lp", 4, False)
        self.upsample2 = block.DySample(512, 2, "lp", 4, False)
        self.neckConv1 = block.C3k2(1024+512, 512, 2, False, 0.5)
        self.neckConv2 = block.C3k2(512+256, 256, 2, False, 0.5)

    def forward(self, x):
        # input shape (B, 3, 384, 1280)
        x_p3 = self.backboneP3(x)
        x_p4 = self.backboneP4(x_p3)
        x_p5 = self.backboneP5(x_p4)
        x_backbone = self.backboneTop(x_p5)
        
        x = self.upsample1(x_backbone) # 11
        x = torch.cat([x_p4, x], dim=1) # 12-cat backbone P4
        x = self.neckConv1(x)  # 13
        x = self.upsample2(x) # 14
        x = torch.cat([x_p3, x], dim=1) # 15-# cat backbone P3
        x_neck = self.neckConv2(x) # 16 (P3/8-small)
        
        return {
            "backbone": x_backbone,         # (B, 1024, 12, 40)
            'neck': x_neck,                 # (B, 256, 48, 160)
            "cls2d": self.cls2d(x_neck),    # (B, cls_num, 48, 160)
            "box2d": self.box2d(x_neck),    # (B, 64, 48, 160)
            "offset3d": self.offset3d(x_neck),
            "size3d": self.size3d(x_neck),
            "heading": self.heading(x_neck),
            "depth": self.depth(x_neck),
        }


if __name__ == "__main__":
    model = model()
    print(model)
