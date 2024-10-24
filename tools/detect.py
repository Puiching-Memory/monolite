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


def detect(model, device, test_loader, visualizer, logger):
    progress_bar = tqdm(
        enumerate(test_loader), dynamic_ncols=True, leave=True, desc="Detection"
    )
    for i, (inputs, targets, data_info) in progress_bar:
        inputs = inputs.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}

        forward_time = time.time_ns()
        outputs = model(inputs)
        forward_time = (time.time_ns() - forward_time) / 1e6  # ms

        if not os.path.exists(visualizer.save_path):
            os.mkdir(visualizer.save_path)

        result = visualizer.decode(inputs, outputs)

        info = {
            "forward_time": forward_time,
        }

        progress_bar.set_postfix(info)
        progress_bar.update()

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite detection script")
    parser.add_argument("--cfg", dest="cfg", help="path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model = importlib.import_module("model").model()
    checkpoint_dict = torch.load(os.path.join(args.cfg, "checkpoint", "model.pth"),map_location=device,weights_only=True)
    model.load_state_dict(checkpoint_dict)
    # model = torch.compile(model) # Not support in windows
    model.eval()
    model = model.to(device)

    # 导入数据集
    data_cfg = importlib.import_module("dataset").data_cfg()
    data_set = importlib.import_module("dataset").data_set(vars(data_cfg))

    # 导入可视化工具
    visualizer = importlib.import_module("visualizer").visualizer()

    # 打印基本信息
    logger.info(
        f"\n{summary(model, input_size=(data_cfg.batch_size,3,384,1280),mode='train',verbose=0)}"
    )
    logger.info(data_set)

    detect(
        model,
        device,
        data_set.test_loader,
        visualizer,
        logger,
    )
