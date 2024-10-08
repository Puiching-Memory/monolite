import sys
sys.path.append(r"C:\workspace\github\monolite")

import torch
import torch.nn as nn
import torch.nn.functional as F

class trainner(object):
    def __init__(self):
        self.epoch = 150
        self.cudnn = True
        self.amp = True
        self.bf16 = False
        self.ddp = False
        self.dp = False
        self.log_interval = 1
        