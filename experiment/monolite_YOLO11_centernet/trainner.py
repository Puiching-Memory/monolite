import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cfg.base import TrainerBase


class trainner(TrainerBase):

    def get_epoch(self):
        return 5

    def get_save_path(self):
        return r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet\checkpoint"

    def get_log_interval(self):
        return 1

    def is_amp(self):
        return True

    def is_cudnn(self):
        return True


if __name__ == "__main__":
    trainer = trainner()
