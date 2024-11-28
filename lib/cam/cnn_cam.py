import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


class module_feature_saver:
    """
    注册钩子，获取中间层的输出
    ---
    init:
        modle: torch.nn.Module, 需要注册的模型
    save_feature:
        module: torch.nn.Module, 当前层
        input: torch.Tensor, 输入
        output: torch.Tensor, 输出
    remove_hook:
        移除注册的钩子

    """

    def __init__(self, modle: torch.nn.Module):
        self.hook: torch.utils.hooks.RemovableHandle = modle.register_forward_hook(
            self.save_feature
        )

    def save_feature(
        self, model: torch.nn.Module, input: tuple[torch.Tensor], output: torch.Tensor
    ):
        """_summary_

        Args:
            model (torch.nn.Module): _description_
            input (tuple[torch.Tensor]): _description_
            output (torch.Tensor): _description_

        忽略Batch维度,保存第一个输入的第一个通道的特征图
        """
        print(
            f"model {model}\n input {len(input)} \n {input[0].shape}\n output {output.shape}"
        )
        self.temp_feature = output[0][0].detach().numpy()
        self.layer_name = model.__class__.__name__

    def remove_hook(self):
        self.hook.remove()


if __name__ == "__main__":
    pass
