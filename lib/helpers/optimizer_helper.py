# import math
# import torch
import torch.optim as optim
# from torch.optim.optimizer import Optimizer
# from torch.distributed.optim import ZeroRedundancyOptimizer
from lib.opti import adai
#import torchopt

def build_optimizer(cfg_optimizer, model,ddp=False):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{'params': biases, 'weight_decay': 0},
                  {'params': weights, 'weight_decay': cfg_optimizer['weight_decay']}]


    if cfg_optimizer['type'] == 'sgd':
        if ddp:
            #optimizer = ZeroRedundancyOptimizer(parameters,optimizer_class=optim.SGD,lr=cfg_optimizer['lr'],momentum=0.9)
            optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
        else:
            optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    elif cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'adamw':
        optimizer = optim.AdamW(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'adai':
        if ddp:
            optimizer = adai.Adai(parameters,lr=cfg_optimizer['lr']*10,weight_decay=5e-4, decoupled=True)
            # optimizer = ZeroRedundancyOptimizer(parameters,optimizer_class=adai.Adai,lr=cfg_optimizer['lr']*10,weight_decay=5e-4, decoupled=True)
        else:
            optimizer = adai.Adai(parameters,lr=cfg_optimizer['lr']*10,weight_decay=5e-4, decoupled=True)
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer

