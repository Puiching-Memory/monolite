import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import importlib
import sys

class config():
    def __init__(self,config_root:str):
        self.root = config_root
        sys.path.append(self.root)

        self.model = self.build_model()

    def build_model(self):
        model = importlib.import_module('model')

        return model.model

    def build_dataset(self):
        pass

    def build_loss(self):
        pass

if __name__ == '__main__':
    cfg = config(r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet")
    print(cfg.model)