from __future__ import annotations

from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader

from .builder import build_dataset


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = cfg["dataset"]
    train_set = build_dataset({
        "name": dataset_cfg["name"],
        "args": {**dataset_cfg.get("args", {}), "split": dataset_cfg.get("train_split", "train")},
    })
    val_set = build_dataset({
        "name": dataset_cfg["name"],
        "args": {**dataset_cfg.get("args", {}), "split": dataset_cfg.get("val_split", "val")},
    })
    test_set = build_dataset({
        "name": dataset_cfg["name"],
        "args": {**dataset_cfg.get("args", {}), "split": dataset_cfg.get("test_split", "test")},
    })

    train_loader_cfg = dataset_cfg.get("train_loader", {})
    val_loader_cfg = dataset_cfg.get("val_loader", {})
    test_loader_cfg = dataset_cfg.get("test_loader", {})

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_loader_cfg.get("batch_size", 8),
        num_workers=train_loader_cfg.get("num_workers", 4),
        shuffle=train_loader_cfg.get("shuffle", True),
        pin_memory=train_loader_cfg.get("pin_memory", True),
        drop_last=train_loader_cfg.get("drop_last", True),
        persistent_workers=train_loader_cfg.get("persistent_workers", True),
        prefetch_factor=train_loader_cfg.get("prefetch_factor", 2),
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=val_loader_cfg.get("batch_size", 1),
        num_workers=val_loader_cfg.get("num_workers", 2),
        shuffle=val_loader_cfg.get("shuffle", False),
        pin_memory=val_loader_cfg.get("pin_memory", False),
        drop_last=val_loader_cfg.get("drop_last", False),
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_loader_cfg.get("batch_size", 1),
        num_workers=test_loader_cfg.get("num_workers", 2),
        shuffle=test_loader_cfg.get("shuffle", False),
        pin_memory=test_loader_cfg.get("pin_memory", False),
        drop_last=test_loader_cfg.get("drop_last", False),
    )
    return train_loader, val_loader, test_loader
