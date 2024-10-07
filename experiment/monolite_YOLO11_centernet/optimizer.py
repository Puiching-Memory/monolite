import sys
sys.path.append(r"C:\workspace\github\monolite")

import torch
import torch.nn as nn
import torch.nn.functional as F

class Optimizer(object):
    def __init__(self):
        self.optimizer = None
        self.lr_scheduler = None
        self.weight_decay = 0.0005
        self.momentum = 0.9
        