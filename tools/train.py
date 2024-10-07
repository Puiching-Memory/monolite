import sys
import os
sys.path.append(r"C:\workspace\github\monolite")
from lib.utils.logger import logger
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import importlib
import argparse

def train(model, device, train_loader, test_loader, optimizer, scheduler, logger):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monolite training script')
    parser.add_argument('--cfg', dest='cfg', help='settings of detection in yaml format')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sys.path.append(args.cfg)
    model = importlib.import_module('model').model()
    data_cfg = importlib.import_module('dataset').data_cfg()
    data_set = importlib.import_module('dataset').data_set(vars(data_cfg))
    
    model = model.to(device)
    
    logger.info(model)
    logger.info(data_set)
    
    #train(model,device,logger)