import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F

class trainner(object):
    def __init__(self):
        self.epoch = 5
        self.cudnn = True
        self.amp = True
        self.ddp = False
        self.dp = False
        self.log_interval = 1
        self.save_path = r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet\checkpoint"
        