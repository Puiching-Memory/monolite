from __future__ import annotations

import torch
from lib.engine.registry import OPTIMIZERS


@OPTIMIZERS.register("adamw")
def build_adamw(params, lr=0.0002, weight_decay=0.01, betas=(0.9, 0.999)):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


@OPTIMIZERS.register("sgd")
def build_sgd(params, lr=0.01, momentum=0.9, weight_decay=0.0005):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
