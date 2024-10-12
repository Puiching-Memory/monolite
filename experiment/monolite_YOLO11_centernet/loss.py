import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class loss(object):
    def __init__(self):
        pass
    
    @staticmethod
    def loss(output:dict[torch.Tensor], target:dict[torch.Tensor])->torch.Tensor:
        loss1 = nn.L1Loss()(output["backbone"], torch.zeros_like(output["backbone"]))
        return loss1