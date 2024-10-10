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
from tqdm import tqdm
from torchinfo import summary


def train(
    model, trainner, device, train_loader, test_loader, optimizer, scheduler, logger
):
    scaler = torch.amp.GradScaler()
    progress_bar = tqdm(
        range(trainner.epoch), dynamic_ncols=True, leave=True, desc="Training"
    )

    for epoch_now in progress_bar:
        for i, (inputs, coord_range, targets, info) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = nn.L1Loss()(
                    outputs["backbone"], torch.zeros_like(outputs["backbone"])
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(
                {
                    "micostep": f"{i}/{len(train_loader)}",
                    "Iuput": tuple(inputs.shape),
                    "backbone": tuple(outputs["backbone"].shape),
                    "neck": tuple(outputs["neck"].shape),
                    "loss": loss.item(),
                }
            )
            
        progress_bar.update()
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(trainner.save_path, "model.pth") )
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
    # model = torch.compile(model) Not support in windows
    model.train()
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

    # 打印基本信息
    logger.info(
        f"\n{summary(model, input_size=(data_cfg.batch_size,3,384,1280),mode='train',verbose=0)}"
    )
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
