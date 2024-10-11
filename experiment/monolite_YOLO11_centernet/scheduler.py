import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

class scheduler(object):
    def __init__(self,optimizer):
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=0,last_epoch=-1)