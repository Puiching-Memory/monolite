import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pickle
import torchvision
from torchvision.transforms import v2
import time
import sys
import cv2
import math

sys.path.append(os.path.abspath("./"))

from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration, Object3d
from lib.utils.metrics import xyxy2xywh, filter_boxes


class KITTI(data.Dataset):
    def __init__(self, root_dir, split, class_map):
        # 数据集信息,分布先验
        self.root_dir = root_dir
        self.class_map = class_map
        self.split = split
        self.max_objects = 50  # 最大物体数

        # 载入数据集
        assert split in ["train", "val", "trainval", "test"]
        split_dir = os.path.join(root_dir, "ImageSets", split + ".txt")
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # 生成子项目路径
        self.data_dir = os.path.join(
            root_dir, "testing" if split == "test" else "training"
        )
        self.image_dir = os.path.join(self.data_dir, "image_2")
        self.depth_dir = os.path.join(self.data_dir, "depth")
        self.calib_dir = os.path.join(self.data_dir, "calib")
        self.label_dir = os.path.join(self.data_dir, "label_2")

        # 图像转换
        self.image_transforms = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(384, 1280)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_image(self, idx) -> torch.Tensor:
        img_file = os.path.join(self.image_dir, f"{idx}.png")
        assert os.path.exists(img_file)
        image = torchvision.io.read_image(img_file)  # (C,H,W)
        raw_image_shape = torch.tensor(image.shape)
        image = self.image_transforms(image)  # 应用变换

        return image, raw_image_shape

    def get_image_numpy(self, idx) -> np.ndarray:
        img_file = os.path.join(self.image_dir, f"{idx}.png")
        assert os.path.exists(img_file)
        image = cv2.imread(img_file)  # (H,W,C)
        return image

    def get_label(self, idx) -> list[Object3d]:
        label_file = os.path.join(self.label_dir, f"{idx}.txt")
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx) -> Calibration:
        calib_file = os.path.join(self.calib_dir, f"{idx}.txt")
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def build_pkl(self, index) -> None:
        if os.path.exists(f"{self.data_dir}/cache/{index}.pkl"):
            # os.remove(f'{cache_path}/{index}.pkl')
            return
            # pass
        data = self.__getitem__(index)
        with open(f"{self.data_dir}/cache/{index}.pkl", "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        dataload_time = time.time_ns()

        # 获取图像、标签、标定信息
        image, raw_image_shape = self.get_image(self.idx_list[index])
        label = self.get_label(self.idx_list[index])
        calib = self.get_calib(self.idx_list[index])
        numpy_image = self.get_image_numpy(self.idx_list[index])

        distance = np.array(
            [
                math.sqrt(
                    math.pow(i.pos[0], 2)
                    + math.pow(i.pos[1], 2)
                    + math.pow(i.pos[2], 2)
                )
                for i in label
            ]
        )

        target = {"distance": distance}

        info = {
            "dataload_time": (time.time_ns() - dataload_time) / 1e6,  # ms
            "image_id": index,
            "raw_image_shape": raw_image_shape,  # (C,H,W)
        }
        return image, target, info


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    dataset = KITTI(
        r"C:\Users\11386\Downloads\kitti3d",
        "trainval",
        {"Car": 0, "Pedestrian": 1, "Cyclist": 2},
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    _collect = []
    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        print(f"inputs: {inputs.shape}")
        for k, v in zip(targets.keys(), targets.values()):
            print(f"output-{k}:{targets[k].shape}")
        print(f"info: {info}")
        _collect.append(targets)
        # if batch_idx == 100:
        #     break

    # 统计距离分布
    distance = np.array([])
    for i in _collect:
        distance = np.concatenate((distance, i["distance"].cpu().numpy().reshape(-1)))
        # distance.append(i['distance'].cpu().numpy()[0])

    distance = distance[distance <= 1732]  # 过滤掉异常值
    print(np.max(distance), np.min(distance), np.mean(distance), np.median(distance))

    # 统计3D框数量分布
    
    plt.hist(distance, bins=100)
    plt.show()
