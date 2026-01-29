from __future__ import annotations

import torch
from lib.engine.registry import SCHEDULERS


@SCHEDULERS.register("cosine")
def build_cosine(optimizer: torch.optim.Optimizer, T_max: int = 50, eta_min: float = 0.0):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


@SCHEDULERS.register("step")
def build_step(optimizer: torch.optim.Optimizer, step_size: int = 20, gamma: float = 0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
