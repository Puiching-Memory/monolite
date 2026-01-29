from __future__ import annotations

import os
import torch
from typing import Any, Dict, Optional


def save_checkpoint(state: Dict[str, Any], filename: str) -> str:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not filename.endswith(".pth"):
        filename = f"{filename}.pth"
    torch.save(state, filename)
    return filename


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    filename: str,
    map_location: str | torch.device = "cpu",
) -> int:
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    checkpoint = torch.load(filename, map_location=map_location)
    epoch = checkpoint.get("epoch", -1)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state", {})))
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return epoch
