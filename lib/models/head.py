import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

import sys
sys.path.append(r"C:\workspace\github\monolite")

from lib.models.block import Conv,DFL,DWConv