import sys
sys.path.append(r"C:\workspace\github\monolite")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class optimizer(object):
    def __init__(self,model):
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999),weight_decay=1e-4,amsgrad=True)