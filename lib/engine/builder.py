from __future__ import annotations

from typing import Any, Dict

import torch

# Import registry items to ensure they are registered before build_* calls.
# These imports are used for side effects only.
import lib.datasets.kitti as _kitti  # noqa: F401
import lib.losses.monolite_loss as _monolite_loss  # noqa: F401
import lib.models.yolo26_monodle as _yolo26_monodle  # noqa: F401
import lib.optim.registry_items as _optim_registry_items  # noqa: F401
import lib.schedulers.registry_items as _scheduler_registry_items  # noqa: F401

from .registry import MODELS, LOSSES, DATASETS, OPTIMIZERS, SCHEDULERS


def _get_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("name") or cfg.get("type") or cfg.get("_name")


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = _get_name(cfg)
    args = cfg.get("args", {})
    return MODELS.get(name)(**args)


def build_loss(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = _get_name(cfg)
    args = cfg.get("args", {})
    return LOSSES.get(name)(**args)


def build_dataset(cfg: Dict[str, Any]):
    name = _get_name(cfg)
    args = cfg.get("args", {})
    return DATASETS.get(name)(**args)


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    name = _get_name(cfg)
    args = cfg.get("args", {})
    return OPTIMIZERS.get(name)(model.parameters(), **args)


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    name = _get_name(cfg)
    args = cfg.get("args", {})
    return SCHEDULERS.get(name)(optimizer, **args)
