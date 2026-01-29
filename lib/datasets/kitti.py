import os
import sys
import time
import pickle
import itertools
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.transforms import v2
from scipy.spatial.transform import Rotation

from lib.engine.registry import DATASETS

# --- Constants ---

COLOUR_MAP = {
    "Pedestrian": (106, 0, 228),
    "Cyclist": (119, 11, 32),
    "Car": (165, 42, 42),
    "Van": (0, 0, 192),
    "Truck": (197, 226, 255),
    "Person_sitting": (0, 60, 100),
    "Tram": (0, 0, 142),
    "Misc": (255, 77, 255),
}

num_heading_bin = 12

# --- Utility Functions (formerly utils.py) ---

def check_range(angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle

def get_angle_from_box3d(box3d_pts_3d):
    direct_vec = (box3d_pts_3d[0] + box3d_pts_3d[1]) / 2 - (
        box3d_pts_3d[2] + box3d_pts_3d[3]
    ) / 2
    if direct_vec[0] >= 0 and direct_vec[-1] >= 0:
        angle = -np.arctan(direct_vec[-1] / direct_vec[0])
    elif direct_vec[0] < 0 and direct_vec[-1] >= 0:
        angle = -(np.pi - np.arctan(np.abs(direct_vec[-1] / direct_vec[0])))
    elif direct_vec[0] < 0 and direct_vec[-1] < 0:
        angle = np.pi - np.arctan(np.abs(direct_vec[-1] / direct_vec[0]))
    elif direct_vec[0] >= 0 and direct_vec[-1] < 0:
        angle = np.arctan(np.abs(direct_vec[-1] / direct_vec[0]))
    return angle

def angle2class(angle):
    """Convert continuous angle to discrete class and residual."""
    angle = angle % (2 * np.pi)
    assert angle >= 0 and angle <= 2 * np.pi
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

def class2angle(cls, residual, to_label_format=False):
    """Inverse function to angle2class."""
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle

def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap

def calculate_lines(num_points, group_size=8):
    if num_points % group_size != 0:
        raise ValueError("The number of points must be a multiple of 8.")
    links = []
    for i in range(num_points // group_size):
        points = list(range(group_size * i, group_size * (i + 1)))
        for pair in itertools.combinations(points, 2):
            links.append(pair)
    return links

# --- KITTI Helper Functions and Classes (formerly kittiUtils.py) ---

def get_objects_from_label(label_path: str) -> list:
    with open(label_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(" ") for line in lines]
        lines = [
            [
                str(line[0]),
                float(line[1]),
                int(line[2]),
                float(line[3]),
                float(line[4]),
                float(line[5]),
                float(line[6]),
                float(line[7]),
                float(line[8]),
                float(line[9]),
                float(line[10]),
                float(line[11]),
                float(line[12]),
                float(line[13]),
                float(line[14]),
            ]
            for line in lines
        ]
    return [Object3d(*line) for line in lines]

@dataclass
class Object3d:
    cls_type: str
    trucation: float
    occlusion: int
    alpha: float
    box2d_x_min: float
    box2d_y_min: float
    box2d_x_max: float
    box2d_y_max: float
    h: float
    w: float
    l: float
    pos_x: float
    pos_y: float
    pos_z: float
    ry: float
    score: float = -1.0

    def __post_init__(self):
        self.box2d = np.array(
            [self.box2d_x_min, self.box2d_y_min, self.box2d_x_max, self.box2d_y_max]
        )
        self.pos = np.array([self.pos_x, self.pos_y, self.pos_z])
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.level_int, self.level_str = self.get_obj_level()

    def get_obj_level(self) -> tuple[int, str]:
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1
        if self.trucation == -1:
            return 0, "DontCare"
        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            return 1, "Easy"
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            return 2, "Moderate"
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            return 3, "Hard"
        else:
            return 4, "UnKnown"

    def generate_corners3d(self):
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        R = np.array(
            [
                [np.cos(self.ry), 0, np.sin(self.ry)],
                [0, 1, 0],
                [-np.sin(self.ry), 0, np.cos(self.ry)],
            ]
        )
        corners3d = np.vstack([x_corners, y_corners, z_corners])
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def generate_label_matrix(self) -> np.ndarray:
        label_matrix = np.hstack(
            [
                Rotation.from_euler("xyz", [0, 0, self.ry]).as_matrix(),
                self.pos.reshape(3, 1),
            ]
        )
        label_matrix = np.vstack([label_matrix, np.array([0, 0, 0, 1])])
        return label_matrix

    def to_kitti_format(self):
        kitti_str = (
            f"{self.cls_type} {self.trucation:.2f} {int(self.occlusion)} "
            f"{self.alpha:.2f} {self.box2d[0]:.2f} {self.box2d[1]:.2f} "
            f"{self.box2d[2]:.2f} {self.box2d[3]:.2f} {self.h:.2f} "
            f"{self.w:.2f} {self.l:.2f} {self.pos[0]:.2f} {self.pos[1]:.2f} "
            f"{self.pos[2]:.2f} {self.ry:.2f}"
        )
        return kitti_str

@dataclass
class Calibration:
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    R0: np.ndarray
    V2C: np.ndarray
    Tr_imu_to_velo: np.ndarray
    
    def __post_init__(self):
        self.C2V = inverse_rigid_trans(self.V2C)
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
        assert self.fu == self.fv, "%.8f != %.8f" % (self.fu, self.fv)

    def scale(self, scale_x, scale_y):
        self.P2[0, :] *= scale_x
        self.P2[1, :] *= scale_y
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def lidar_to_rect(self, pts_lidar):
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect):
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_2d_hom[:, 2]).T
        pts_rect_depth = pts_2d_hom[:, 2]
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1
        )
        return pts_rect

    def depthmap_to_rect(self, depth_map: np.ndarray):
        y_idxs, x_idxs = np.indices(depth_map.shape)
        x_idxs = x_idxs.ravel()
        y_idxs = y_idxs.ravel()
        depth = depth_map.ravel()
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2
        )
        img_pts = np.matmul(corners3d_hom, self.P2.T)
        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)
        boxes = np.concatenate(
            (
                x1.reshape(-1, 1),
                y1.reshape(-1, 1),
                x2.reshape(-1, 1),
                y2.reshape(-1, 1),
            ),
            axis=1,
        )
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2
        )
        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)
        return np.array([x, y, z])

    def alpha2ry(self, alpha, u):
        ry = alpha + np.arctan2(u - self.cu, self.fu)
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)
        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi
        return alpha

    def cart_to_hom(self, pts: np.ndarray) -> np.ndarray:
        return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))

def get_calib_from_file(calib_path: str) -> Calibration:
    with open(calib_path) as f:
        lines = f.readlines()
    matrices = [
        np.array(lines[i].strip().split(" ")[1:], dtype=np.float32) for i in range(7)
    ]
    P0 = matrices[0].reshape(3, 4)
    P1 = matrices[1].reshape(3, 4)
    P2 = matrices[2].reshape(3, 4)
    R0 = matrices[4].reshape(3, 3)
    Tr_velo_to_cam = matrices[5].reshape(3, 4)
    Tr_imu_to_velo = matrices[6].reshape(3, 4)
    return Calibration(P0, P1, P2, R0, Tr_velo_to_cam, Tr_imu_to_velo)

def inverse_rigid_trans(Tr: np.ndarray) -> np.ndarray:
    inv_Tr = np.zeros_like(Tr)
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# --- KITTI Dataset Class ---

@DATASETS.register("kitti")
class KITTI(data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        class_map,
        image_size=(384, 1248),
        max_objects: int = 50,
        return_boxes2d: bool = True,
        downsample: int = 8,
        use_3d_center: bool = True,
        use_mean_size: bool = False,
    ):
        self.root_dir = root_dir
        self.class_map = class_map
        self.split = split
        self.max_objects = max_objects
        self.return_boxes2d = return_boxes2d
        self.image_size = image_size
        self.downsample = downsample
        self.use_3d_center = use_3d_center
        self.num_classes = len(self.class_map)

        mean_size_map = {
            "Pedestrian": np.array([1.76255119, 0.66068622, 0.84422524], dtype=np.float32),
            "Car": np.array([1.52563191, 1.62856739, 3.52588311], dtype=np.float32),
            "Cyclist": np.array([1.73698127, 0.59706367, 1.76282397], dtype=np.float32),
        }
        self.cls_mean_size = np.zeros((self.num_classes, 3), dtype=np.float32)
        if use_mean_size:
            for cls_name, cls_id in self.class_map.items():
                if cls_name in mean_size_map:
                    self.cls_mean_size[cls_id] = mean_size_map[cls_name]

        assert split in ["train", "val", "trainval", "test"]
        split_dir = os.path.join(root_dir, "ImageSets", split + ".txt")
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        self.data_dir = os.path.join(
            root_dir, "testing" if split == "test" else "training"
        )
        self.image_dir = os.path.join(self.data_dir, "image_2")
        self.calib_dir = os.path.join(self.data_dir, "calib")
        self.label_dir = os.path.join(self.data_dir, "label_2")

        self.image_transforms = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.image_size[0], self.image_size[1])),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.image_size[0] % self.downsample != 0 or self.image_size[1] % self.downsample != 0:
            raise ValueError("image_size must be divisible by downsample")

    def get_image_torch(self, index_string: str) -> torch.Tensor:
        img_file = os.path.join(self.image_dir, f"{index_string}.png")
        assert os.path.exists(img_file)
        image = torchvision.io.decode_image(img_file)
        raw_image_shape = torch.tensor(image.shape)
        image = self.image_transforms(image)
        return image, raw_image_shape

    def get_labels(self, index_string: str) -> list[Object3d]:
        label_file = os.path.join(self.label_dir, f"{index_string}.txt")
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, index_string) -> Calibration:
        calib_file = os.path.join(self.calib_dir, f"{index_string}.txt")
        assert os.path.exists(calib_file)
        return get_calib_from_file(calib_file)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        dataload_time = time.perf_counter_ns()
        index_string = self.idx_list[index]
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            future_image = executor.submit(self.get_image_torch, index_string)
            future_labels = executor.submit(self.get_labels, index_string)
            future_calib = executor.submit(self.get_calib, index_string)
            calib = future_calib.result()
            labels = future_labels.result()
            image, raw_image_shape = future_image.result()

        raw_h = int(raw_image_shape[1])
        raw_w = int(raw_image_shape[2])
        scale_y = self.image_size[0] / raw_h
        scale_x = self.image_size[1] / raw_w
        calib.scale(scale_x, scale_y)

        target = self._build_3d_targets(labels, calib, raw_image_shape)
        if self.return_boxes2d:
            boxes2d, labels_ids = self._build_2d_targets(labels, raw_image_shape)
            target["boxes2d"] = boxes2d
            target["labels"] = labels_ids

        target["image_size"] = torch.tensor(self.image_size, dtype=torch.float32)

        info = {
            "dataload_time": (time.perf_counter_ns() - dataload_time) / 1e6,
            "image_id": index,
            "raw_image_shape": raw_image_shape,
            "calib": calib.P2,
        }
        return image, target, info

    def _build_3d_targets(
        self,
        labels: list[Object3d],
        calib: Calibration,
        raw_image_shape: torch.Tensor,
    ) -> dict:
        raw_h = int(raw_image_shape[1])
        raw_w = int(raw_image_shape[2])
        scale_y = self.image_size[0] / raw_h
        scale_x = self.image_size[1] / raw_w

        features_w = self.image_size[1] // self.downsample
        features_h = self.image_size[0] // self.downsample
        features_size = (features_w, features_h)

        heatmap = np.zeros((self.num_classes, features_h, features_w), dtype=np.float32)
        depth = np.zeros((self.max_objects, 1), dtype=np.float32)
        heading_bin_arr = np.zeros((self.max_objects, 1), dtype=np.int64)
        heading_res_arr = np.zeros((self.max_objects, 1), dtype=np.float32)
        size_3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        offset_3d = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices = np.zeros((self.max_objects), dtype=np.int64)
        mask_3d = np.zeros((self.max_objects), dtype=np.uint8)

        object_num = min(len(labels), self.max_objects)
        for i in range(object_num):
            obj = labels[i]
            if obj.cls_type not in self.class_map:
                continue

            bbox_2d = obj.box2d.copy()
            bbox_2d[0] *= scale_x
            bbox_2d[2] *= scale_x
            bbox_2d[1] *= scale_y
            bbox_2d[3] *= scale_y
            bbox_2d /= self.downsample

            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                dtype=np.float32,
            )

            center_3d_pts = obj.pos + [0, -obj.h / 2, 0]
            center_3d_pts = center_3d_pts.reshape(-1, 3)
            center_3d_pts, _ = calib.rect_to_img(center_3d_pts)
            center_3d_pts = center_3d_pts[0]
            center_3d_pts /= self.downsample

            center_heatmap = (
                center_3d_pts.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
            )
            if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]:
                continue
            if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]:
                continue

            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            cls_id = self.class_map[obj.cls_type]
            draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

            indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
            depth[i] = obj.pos[-1]
            heading_bin_arr[i], heading_res_arr[i] = angle2class(obj.alpha)
            offset_3d[i] = center_3d_pts - center_heatmap
            size_3d[i] = np.array([obj.h, obj.w, obj.l], dtype=np.float32) - self.cls_mean_size[cls_id]
            mask_3d[i] = 1

        return {
            "heatmap3D": heatmap,
            "depth": depth,
            "heading_bin": heading_bin_arr,
            "heading_res": heading_res_arr,
            "size_3d": size_3d,
            "offset_3d": offset_3d,
            "indices": indices,
            "mask_3d": mask_3d,
        }

    def _build_2d_targets(self, labels: list[Object3d], raw_image_shape: torch.Tensor):
        raw_h = int(raw_image_shape[1])
        raw_w = int(raw_image_shape[2])
        scale_y = self.image_size[0] / raw_h
        scale_x = self.image_size[1] / raw_w
        boxes = []
        cls_ids = []
        for obj in labels:
            if obj.cls_type not in self.class_map:
                continue
            x1, y1, x2, y2 = obj.box2d
            boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            cls_ids.append(self.class_map[obj.cls_type])

        if len(boxes) == 0:
            return torch.zeros((self.max_objects, 4)), torch.full((self.max_objects,), -1, dtype=torch.int64)

        boxes_tensor = torch.tensor(boxes[:self.max_objects], dtype=torch.float32)
        labels_tensor = torch.tensor(cls_ids[:self.max_objects], dtype=torch.int64)
        if boxes_tensor.shape[0] < self.max_objects:
            pad = self.max_objects - boxes_tensor.shape[0]
            boxes_tensor = torch.cat([boxes_tensor, torch.zeros((pad, 4))], dim=0)
            labels_tensor = torch.cat([labels_tensor, torch.full((pad,), -1, dtype=torch.int64)], dim=0)
        return boxes_tensor, labels_tensor

if __name__ == "__main__":
    # Test block
    pass
