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

from lib.datasets.kittiUtils import get_objects_from_label,get_calib_from_file
from lib.datasets.kittiUtils import Calibration, Object3d
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
        
        # 生成3D锚点
        _calib = self.get_calib(self.idx_list[0])
        _numpy_image = self.get_image_numpy(self.idx_list[0])
        self.anchor3D = [
            _calib.camera_dis_to_rect(x, y, depth)
            for x in range(0, _numpy_image.shape[1], 8)
            for y in range(0, _numpy_image.shape[0], 8)
            for depth in range(1, 80)
        ]
        self.anchor3D = np.array(self.anchor3D).reshape((-1, 3))

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
        raw_image_shape = np.array(image.shape)
        image = cv2.resize(image, (1280, 384))  # (H,W,C)
        return image, raw_image_shape

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

    def build_parquet(self, index) -> str:
        import pyarrow.parquet as parquet
        import pyarrow.feather as feather
        import pyarrow as pa
        import os

        tables = []
        for i in []:
            table = pa.Table.from_arrays(
                [
                    matrix.flatten(),
                ],
                names=["data1"],
            )

            tables.append(table)

        # feather.write_feather(pa.concat_tables(tables), "output.feather")
        parquet.write_table(pa.concat_tables(tables), "output.parquet")

        return "output.parquet"
    
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        dataload_time = time.perf_counter_ns()

        # 获取图像、标签、标定信息
        image, raw_image_shape = self.get_image(self.idx_list[index])
        label = self.get_label(self.idx_list[index])
        calib = self.get_calib(self.idx_list[index])
        numpy_image = self.get_image_numpy(self.idx_list[index])


        target = {
            "heatmap3D": heatmap3D,
        }

        info = {
            "dataload_time": (time.perf_counter_ns() - dataload_time) / 1e6,  # ms
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
