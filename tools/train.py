import sys
import os

sys.path.append(os.path.abspath("./"))

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
import time
import swanlab
import datetime


def train(
    model, trainner, device, train_loader, test_loader, optimizer, scheduler, loss_fn, logger
):
    scaler = torch.amp.GradScaler()
    progress_bar = tqdm(
        range(trainner.epoch), dynamic_ncols=True, leave=True, desc="Training"
    )

    for epoch_now in progress_bar:
        for i, (inputs, targets, data_info) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                forward_time = time.time_ns()
                outputs = model(inputs)
                forward_time = (time.time_ns() - forward_time) / 1e6  # ms

                loss_time = time.time_ns()
                loss,loss_info = loss_fn(outputs, targets)
                loss_time = (time.time_ns() - loss_time) / 1e6  # ms

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            info = {
                "epoch": epoch_now,
                "micostep": i,
                "allstep": len(train_loader),
                "forward_time(ms)": forward_time,
                "loss_time(ms)": loss_time,
                "dataload_time(ms)": torch.mean(info["dataload_time"]).item(),
                "loss": loss.item(),
                **loss_info,
            }
            progress_bar.set_postfix(info)
            swanlab.log(info)
            # logger.info(f"input_shape: {tuple(inputs.shape)} backbone: {tuple(outputs['backbone'].shape)} neck: {tuple(outputs['neck'].shape)} box2d: {tuple(outputs['box2d'].shape)}")

        progress_bar.update()
        scheduler.step()
        if not os.path.exists(trainner.save_path):
            os.mkdir(trainner.save_path)
        torch.save(model.state_dict(), os.path.join(trainner.save_path, "model.pth"))
        logger.info(f"checkpoint: {epoch_now+1} saved to {trainner.save_path}")

        #break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite training script")
    parser.add_argument(
        "--cfg", dest="cfg", help="path to config file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model = importlib.import_module("model").model()
    #model = torch.compile(model) # Not support in windows
    model.train()
    model = model.to(device)

    # 导入数据集
    data_cfg = importlib.import_module("dataset").data_cfg()
    data_set = importlib.import_module("dataset").data_set(vars(data_cfg))

    # 导入优化器
    optimizer = importlib.import_module("optimizer").optimizer(model).optimizer

    # 导入学习率衰减器
    scheduler = importlib.import_module("scheduler").scheduler(optimizer).scheduler
    
    # 导入损失函数
    loss_fn = importlib.import_module("loss").loss(device).loss

    # 导入训练配置
    trainner = importlib.import_module("trainner").trainner()

    # 打印基本信息
    logger.info(
        f"\n{summary(model, input_size=(data_cfg.batch_size,3,384,1280),mode='train',verbose=0)}"
    )
    logger.info(data_set)
    logger.info(optimizer)
    logger.info(scheduler)

    # 初始化swanlab,启动$swanlab watch ./logs
    swanlab.init(
        experiment_name=f"{os.path.basename(args.cfg)}_{datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')}",
        logdir="./logs",
        mode="local",
    )

    train(
        model,
        trainner,
        device, 
        data_set.train_loader,
        data_set.test_loader,
        optimizer,
        scheduler,
        loss_fn,
        logger,
    )
