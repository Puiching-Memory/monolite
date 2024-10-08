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
import torch.amp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import importlib
import argparse


def train(
    model, trainner, device, train_loader, test_loader, optimizer, scheduler, logger
):
    scaler = torch.amp.GradScaler()
    
    for epoch_now in range(trainner.epoch):
        model.train()
        for i, (inputs, coord_range, targets, info) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            logger.info(f"Epoch: {epoch_now}/{trainner.epoch} Iter: {i}/{len(train_loader)} Input: {inputs.shape}")
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                outputs = model(inputs)
                loss = nn.L1Loss()(outputs, torch.zeros_like(outputs))
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite training script")
    parser.add_argument(
        "--cfg", dest="cfg", help="settings of detection in yaml format"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model = importlib.import_module("model").model()
    model = model.to(device)

    # 导入数据集
    data_cfg = importlib.import_module("dataset").data_cfg()
    data_set = importlib.import_module("dataset").data_set(vars(data_cfg))

    # 导入优化器
    optimizer = importlib.import_module("optimizer").optimizer(model).optimizer

    # 导入学习率衰减器
    scheduler = importlib.import_module("scheduler").scheduler(optimizer).scheduler

    # 导入训练配置
    trainner = importlib.import_module("trainner").trainner()

    logger.info(model)
    logger.info(data_set)
    logger.info(optimizer)
    logger.info(scheduler)

    train(
        model,
        trainner,
        device,
        data_set.train_loader,
        data_set.test_loader,
        optimizer,
        scheduler,
        logger,
    )
