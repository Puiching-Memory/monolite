import sys
import os

sys.path.append(os.path.abspath("./"))

import argparse
import random
import numpy as np
import torch


from lib.engine import (
    load_config,
    build_model,
    build_loss,
    build_optimizer,
    build_scheduler,
    build_dataloaders,
    Trainer,
    init_logger,
    load_checkpoint,
)

try:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
except Exception:
    local_rank = -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite training script")
    parser.add_argument("--cfg", dest="cfg", required=True, help="path to config file")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    output_dir = cfg.get("output_dir", "./runs/exp")
    logger = init_logger(output_dir)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = build_model(cfg["model"]).to(device)
    
    # Load pretrained backbone if specified
    pretrained = cfg.get("model", {}).get("args", {}).get("pretrained")
    if pretrained and not cfg.get("resume"):
        logger.info(f"Loading pretrained backbone from {pretrained}")
        model.load_pretrained_backbone(pretrained, strict=False)
    
    loss_fn = build_loss(cfg["loss"]).to(device)
    optimizer = build_optimizer(cfg["optimizer"], model)
    scheduler = build_scheduler(cfg["scheduler"], optimizer)

    resume = cfg.get("resume", None)
    if resume:
        logger.info(f"resume from {resume}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, resume, map_location=device)
        cfg.setdefault("trainer", {})["start_epoch"] = start_epoch

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        device=device,
    )
    trainer.train()
