import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class optimizer(object):
    def __init__(self,model):
        self.optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        #self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999),weight_decay=1e-4,amsgrad=True)