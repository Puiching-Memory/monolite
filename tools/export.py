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
from onnxsim import simplify
import onnx


def export(model, device,test_loader, logger):
    for inputs, targets, data_info in test_loader:
        inputs = inputs.to(device)
        with open(os.path.join(args.cfg, "checkpoint", "model.onnx"), "wb") as f:
            onnx_program = torch.onnx.export(
                model,
                (inputs,),
                f,
                verbose=False,
                dynamo=False,
                opset_version=20,
            )
        break

    model = onnx.load(os.path.join(args.cfg, "checkpoint", "model.onnx"))
    onnx.checker.check_model(model)
    # TODO:not support opset > 20 yet
    model_simp, check = simplify(model) 
    onnx.save(model_simp, os.path.join(args.cfg, "checkpoint", "model_simp.onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite detection script")
    parser.add_argument("--cfg", dest="cfg", help="path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model = importlib.import_module("model").model()
    model = model.to(device)
    
    checkpoint_dict = torch.load(
        os.path.join(args.cfg, "checkpoint", "model.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint_dict)
    # model = torch.compile(model) # Not support in windows
    # model.eval()
    
    # 导入数据集
    data_cfg = importlib.import_module("dataset").data_cfg()
    data_set = importlib.import_module("dataset").data_set(vars(data_cfg))

    # 打印基本信息
    logger.info(
        f"\n{summary(model, input_size=(data_cfg.batch_size,3,384,1280),mode='train',verbose=0)}"
    )
    logger.info(data_set)

    export(
        model,
        device,
        data_set.test_loader,
        logger,
    )
