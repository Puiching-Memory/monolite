import sys
import os

import torch.nn.intrinsic

sys.path.append(os.path.abspath("./"))

from lib.utils.logger import logger, build_progress
from lib.models.init import weight_init

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import importlib
import argparse
from rich.live import Live
from torchinfo import summary
import time
import swanlab
import datetime
import psutil

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

pid = os.getpid()
pcontext =  psutil.Process(pid)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 设置同步cuda,仅debug时使用


def train(
    model,
    trainner,
    device,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    loss_fn,
    visualizer,
    logger,
):
    scaler = torch.amp.GradScaler()

    table, progress, task_ids = build_progress(len(train_loader), trainner.epoch)
    with Live(table, refresh_per_second=10) as live:
        for epoch_now in range(trainner.epoch):
            model.train()
            for i, (inputs, targets, data_info) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                targets = {key: value.to(device) for key, value in targets.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    forward_time = time.time_ns()
                    outputs = model(inputs)
                    forward_time = (time.time_ns() - forward_time) / 1e6  # ms

                    loss_time = time.time_ns()
                    loss, loss_info = loss_fn(outputs, targets)
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
                    "dataload_time(ms)": round(torch.mean(data_info["dataload_time"]).item(),4),
                    "loss": round(loss.item(),2),
                    "cpu":round(pcontext.cpu_percent(),2),
                    "ram":round(pcontext.memory_percent(),2),
                    **loss_info,
                }
                swanlab.log(info)
                
                progress["Progress"].update(task_ids["jobId_microstep"],completed=info["micostep"],total=info["allstep"])
                progress["Info"].update(task_ids["jobId_microstep_info"],completed=info["micostep"],total=info["allstep"])
                progress["Time"].update(task_ids["jobId_datatime_info"],completed=info["dataload_time(ms)"])
                progress["Time"].update(task_ids["jobId_losstime_info"],completed=info["loss_time(ms)"])
                progress["Time"].update(task_ids["jobId_forwardtime_info"],completed=info["forward_time(ms)"])
                progress["Loss"].update(task_ids["jobId_loss_info"],completed=info["loss"])
                progress["System"].update(task_ids["jobId_cpu_info"],completed=info["cpu"])
                progress["System"].update(task_ids["jobId_ram_info"],completed=info["ram"])
                
            scheduler.step()

            # 保存模型
            if not os.path.exists(trainner.save_path):
                os.mkdir(trainner.save_path)
            torch.save(
                model.state_dict(), os.path.join(trainner.save_path, "model.pth")
            )
            logger.info(f"checkpoint: {epoch_now+1} saved to {trainner.save_path}")

            # 保存模型预测可视化结果
            model.eval()
            results = visualizer.decode_output(inputs, outputs)
            results = {key: swanlab.Image(value) for key, value in results.items()}
            swanlab.log(results)

            # 保存真值可视化结果
            results = visualizer.decode_target(inputs, targets)
            results = {key: swanlab.Image(value) for key, value in results.items()}
            swanlab.log(results)
            
            progress["Progress"].update(task_ids["jobId_all"],completed=epoch_now+1,total=trainner.epoch)
            progress["Info"].update(task_ids["jobId_epoch_info"],completed=epoch_now+1,total=trainner.epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite training script")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        default=r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet",
        help="path to config file",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model = importlib.import_module("model").model()
    # model = torch.compile(model) # Not support in windows
    model.apply(weight_init)
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

    # 导入可视化工具
    visualizer = importlib.import_module("visualizer").visualizer()

    # 打印基本信息
    print(
        f"\n{summary(model, input_size=(data_cfg.batch_size,3,384,1280),mode='train',verbose=0,depth=2)}"
    )
    logger.info(data_set)
    logger.info(optimizer)
    logger.info(scheduler)

    # 初始化swanlab,启动$swanlab watch ./logs
    swanlab.init(
        project="monolite",
        experiment_name=f"{os.path.basename(args.cfg)}_{datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')}",
        # logdir="./logs", # 本地模式
        # mode="local",
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
        visualizer,
        logger,
    )
