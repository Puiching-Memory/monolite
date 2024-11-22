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
from lib.utils.metrics import xyxy2xywh, clip_coordinates


class KITTI(data.Dataset):
    def __init__(self, root_dir, split, class_map):
        # 数据集信息,分布先验
        self.root_dir = root_dir
        self.class_map = class_map
        self.split = split

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
        image = self.image_transforms(image)  # 应用变换
        return image

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
        image = self.get_image(self.idx_list[index])
        label = self.get_label(self.idx_list[index])
        calib = self.get_calib(self.idx_list[index])
        numpy_image = self.get_image_numpy(self.idx_list[index])

        # 生成heatmap
        heatmap = np.zeros(image.shape[1:3])

        # 每行label都为一个Object3d对象，调用generate_corners3d()方法生成3D坐标
        corners_ego3d = np.array([i.generate_corners3d() for i in label])  # (N,8,3)
        # 转换为图像坐标系下的2D坐标
        boxes_image2d, corners_image2d = calib.corners3d_to_img_boxes(
            corners_ego3d
        )  # (N,4) (N,8,2)

        boxes_image2d = xyxy2xywh(boxes_image2d).astype(
            np.int32
        )  # 转换为xywh格式, (N,4)
        boxes_image2d[:, [0, 1]] = boxes_image2d[:, [1, 0]]  # 坐标顺序转换, (N,4)

        boxes_image2d = clip_coordinates(
            boxes_image2d, heatmap.shape[1], heatmap.shape[0]
        )  # 裁剪超出边界的框
        for x, y, w, h in boxes_image2d:
            cv2.circle(heatmap, (y, x), 13, (255), -1)

        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)  # 高斯模糊
        heatmap = cv2.resize(
            heatmap, (int(1280 / 8), int(384 / 8)), interpolation=cv2.INTER_AREA
        )  # 缩放至heatmap大小
        heatmap = np.expand_dims(heatmap, axis=0)  # 增加维度
        
        heatmap_backgroud = np.ones_like(heatmap)  # heatmap背景
        heatmap_backgroud = heatmap_backgroud - heatmap

        heatmap = np.concatenate(
            (heatmap_backgroud, heatmap), axis=0
        )  # 合并heatmap和背景

        # numpy_image[boxes_image2d[:,0],boxes_image2d[:,1]] = (0,0,255)

        # for x1,y1,x2,y2 in boxes_image2d: # 在图像上绘制2D框
        #     cv2.rectangle(numpy_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # for point in corners_image2d: # 在图像上绘制3D框
        #     for x,y in point:
        #         cv2.circle(numpy_image, (int(x), int(y)), 3, (0, 0, 255), -1)
        # cv2.imwrite("temp.png", heatmap)

        # cls_type = [i.cls_type for i in label]  # 获取类别标签
        # cls_type = [i for i in cls_type if i in self.cfg["writelist"]]  # 筛选类别标签

        target = {
            "cls2d": 0,
            "heatmap": heatmap,
            "offset3d": 0,
            "size3d": 0,
            "heading": 0,
            "depth": 0,
        }

        info = {
            "dataload_time": (time.time_ns() - dataload_time) / 1e6,  # ms
            "image_id": index,
        }
        return image, target, info


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    cfg = {"split": "trainval", "writelist": ["Car", "Pedestrian", "Cyclist"]}
    dataset = KITTI(r"C:\Users\11386\Downloads\kitti3d", cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        print(inputs.shape, targets["heatmap"].shape, info)
        break
