import sys
sys.path.append(r"C:\workspace\github\monolite")

import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super().__init__()
        from lib.models import block

        self.layer = nn.Sequential(
            block.Conv(3,64,3,2),               # 0-P1/2
            block.Conv(64,128,3,2),              # 1-P2/4
            block.C3k2(128,256,2,False,0.25),   # 2
            block.Conv(256,256,3,2),            # 3-P3/8
            block.C3k2(256,512,2,False,0.25),   # 4
            block.Conv(512,512,3,2),            # 5-P4/16
            block.C3k2(512,512,2,True,0.5),     # 6
            block.Conv(512,1024,3,2),           # 7-P5/32
            block.C3k2(1024,1024,2,True,0.5),   # 8
            block.SPPF(1024,1024,5),            # 9
            block.C3k2(1024,512,2,False,0.5),   # 10-head
        )

    def forward(self, x):
        return self.layer(x)
    

if __name__ == "__main__":
    model = model()
    print(model)
