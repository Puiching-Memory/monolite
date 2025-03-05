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
from distance3d.containment_test import points_in_box
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.append(os.path.abspath("./"))

from lib.datasets.kittiUtils import get_objects_from_label, get_calib_from_file
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
        self.anchor3D = [
            _calib.camera_dis_to_rect(x, y, depth)
            for x in range(0, 1280, 8)
            for y in range(0, 384, 8)
            for depth in range(1, 80, 1)
        ]
        self.anchor3D = np.array(self.anchor3D).reshape((-1, 3))
        print(self.anchor3D.shape)

    def get_image_torch(self, index_string: str) -> torch.Tensor:
        img_file = os.path.join(self.image_dir, f"{index_string}.png")
        assert os.path.exists(img_file)
        image = torchvision.io.decode_image(img_file)  # (C,H,W)
        raw_image_shape = torch.tensor(image.shape)
        image = self.image_transforms(image)  # 应用变换

        return image, raw_image_shape

    def get_image_numpy(self, index_string: str) -> np.ndarray:
        img_file = os.path.join(self.image_dir, f"{index_string}.png")
        assert os.path.exists(img_file)
        image = cv2.imread(img_file)  # (H,W,C)
        raw_image_shape = np.array(image.shape)
        image = cv2.resize(image, (1280, 384))  # (H,W,C)
        return image, raw_image_shape

    def get_labels(self, index_string: str) -> list[Object3d]:
        label_file = os.path.join(self.label_dir, f"{index_string}.txt")
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, index_string) -> Calibration:
        calib_file = os.path.join(self.calib_dir, f"{index_string}.txt")
        assert os.path.exists(calib_file)
        return get_calib_from_file(calib_file)

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
        index_string = self.idx_list[index]

        ### 获取图像、标签、标定信息

        # 并行执行
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            # 提交任务
            future_image = executor.submit(self.get_image_torch, index_string)
            future_labels = executor.submit(self.get_labels, index_string)
            future_calib = executor.submit(self.get_calib, index_string)

            # 获取结果
            calib = future_calib.result()
            labels = future_labels.result()
            image, raw_image_shape = future_image.result()

        # 顺序执行
        # image, raw_image_shape = self.get_image(index_string)
        # labels = self.get_labels(index_string)
        # calib = self.get_calib(index_string)
        # numpy_image,numpy_image_shape = self.get_image_numpy(self.idx_list[index])

        anchor3D_mask = np.zeros(self.anchor3D.shape[0], dtype=bool)
        
        # FIXME: 当内存不足时,发生异常
        # mask_future = []
        # with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        #     for label in labels:
        #         mask_future.append(
        #             executor.submit(
        #                 points_in_box,
        #                 self.anchor3D,
        #                 label.generate_label_matrix(),
        #                 np.array([label.l, label.w, label.h]),
        #             )
        #         )

        #     for f in mask_future:
        #         _mask = f.result()
        #         anchor3D_mask = anchor3D_mask | _mask

        for label in labels:
            _mask = points_in_box(
                self.anchor3D,
                label.generate_label_matrix(),
                np.array([label.l, label.w, label.h]),
            )
            anchor3D_mask = anchor3D_mask | _mask

        target = {
            "heatmap3D": anchor3D_mask.astype(np.float32),
        }

        info = {
            "dataload_time": (time.perf_counter_ns() - dataload_time) / 1e6,  # ms
            "image_id": index,
            "raw_image_shape": raw_image_shape,  # (C,H,W)
        }
        return image, target, info


if __name__ == "__main__":
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    from torch.utils.data import DataLoader

    dataset = KITTI(
        r"C:\Users\11386\Downloads\kitti3d",
        "trainval",
        {"Car": 0, "Pedestrian": 1, "Cyclist": 2},
    )
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True,num_workers=4,pin_memory=True)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        print(f"inputs: {inputs.shape}")
        for k, v in zip(targets.keys(), targets.values()):
            print(f"output-{k}:{targets[k].shape}")
        print(f"info: {info}")

        break

    profiler.stop()
    profiler.print()

    with open("profiler.html", "w") as f:
        f.write(profiler.output_html())
