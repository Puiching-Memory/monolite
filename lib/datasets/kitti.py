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

        # 初始化
        anchor3d = np.linspace(1, 50, 50)
        anchor3d = anchor3d[np.newaxis, np.newaxis, :]
        anchor3d = np.repeat(anchor3d, 160, axis=0)
        anchor3d = np.repeat(anchor3d, 48, axis=1)

        blk = np.zeros(numpy_image.shape, np.uint8)
        anchor2d = []
        for x in range(0, numpy_image.shape[1], 8):
            for y in range(0, numpy_image.shape[0], 8):
                for depth in range(1, 50, 1):
                    anchor2d.append(calib.camera_dis_to_rect(x, y, depth)[0])
        anchor2d = calib.rect_to_lidar(np.array(anchor2d))
        anchor2d, _ = calib.lidar_to_img(anchor2d)
        # anchor2d,_ = calib.rect_to_img(np.array(anchor2d))
        print(anchor2d)

        for x, y in anchor2d:
            print(x, y)
            # _temp = np.zeros(numpy_image.shape, np.uint8)
            cv2.circle(blk, (int(x), int(y)), 1, (0, 0, 255), -1)
            # blk = cv2.addWeighted(blk, 1.0, _temp, 0.5, 1)

        cv2.addWeighted(numpy_image, 0.5, blk, 0.5, 1)
        cv2.imwrite(f"1.jpg", blk)

        # 每行label都为一个Object3d对象
        corners_ego3d = np.array([i.generate_corners3d() for i in label])  # (N,8,3)
        corners_cam2d = np.array([i.box2d for i in label])  # (N, 4)

        _temp = np.zeros((self.max_objects, 4))
        _temp[: len(corners_cam2d)] = corners_cam2d
        corners_cam2d = _temp

        target = {
            "box2d": corners_cam2d,
        }

        info = {
            "dataload_time": (time.time_ns() - dataload_time) / 1e6,  # ms
            "image_id": index,
            "raw_image_shape": raw_image_shape,  # (C,H,W)
        }
        return image, target, info


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = KITTI(
        r"C:\Users\11386\Downloads\kitti3d",
        "trainval",
        {"Car": 0, "Pedestrian": 1, "Cyclist": 2},
    )
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        print(f"inputs: {inputs.shape}")
        for k, v in zip(targets.keys(), targets.values()):
            print(f"output-{k}:{targets[k].shape}")
        print(f"info: {info}")

        break
