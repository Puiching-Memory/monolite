import numpy as np
import numba
from numba.experimental import jitclass
import cv2
from typing import Union, Optional, Dict

METAINFO = {
    "classes": (
        "Pedestrian",
        "Cyclist",
        "Car",
        "Van",
        "Truck",
        "Person_sitting",
        "Tram",
        "Misc",
    ),
    "palette": [
        (106, 0, 228),
        (119, 11, 32),
        (165, 42, 42),
        (0, 0, 192),
        (197, 226, 255),
        (0, 60, 100),
        (0, 0, 142),
        (255, 77, 255),
    ],
}

################  Object3D  ##################


def get_objects_from_label(label_path: str) -> list:
    with open(label_path, "r") as f:
        lines = f.readlines()
    return [Object3d(line) for line in lines]


class Object3d(object):
    def __init__(self, line: str):
        label = line.strip().split(" ")
        self.cls_type = label[0]
        self.trucation = float(label[1])
        self.occlusion = float(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array(
            (float(label[4]), float(label[5]), float(label[6]), float(label[7])),
            dtype=np.float32,
        )
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array(
            (float(label[11]), float(label[12]), float(label[13])), dtype=np.float32
        )
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        self.score = float(label[15]) if len(label) == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = "DontCare"
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = "Easy"
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = "Moderate"
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = "Hard"
            return 3  # Hard
        else:
            self.level_str = "UnKnown"
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
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
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos

        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            box2d[:, 1] = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(
                    np.int32
                )
            )
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            cv = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            )
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = (
            f"{self.cls_type} {self.trucation:.3f} {self.occlusion:.3f} {self.alpha:.3f} "
            f"box2d: {self.box2d} hwl: [{self.h:.3f} {self.w:.3f} {self.l:.3f}] "
            f"pos: {self.pos} ry: {self.ry:.3f}"
        )
        return print_str

    def to_kitti_format(self):
        kitti_str = (
            f"{self.cls_type} {self.trucation:.2f} {int(self.occlusion)} "
            f"{self.alpha:.2f} {self.box2d[0]:.2f} {self.box2d[1]:.2f} "
            f"{self.box2d[2]:.2f} {self.box2d[3]:.2f} {self.h:.2f} "
            f"{self.w:.2f} {self.l:.2f} {self.pos[0]:.2f} {self.pos[1]:.2f} "
            f"{self.pos[2]:.2f} {self.ry:.2f}"
        )
        return kitti_str


###################  calibration  ###################
def get_calib_from_file(calib_file: str) -> Dict[str, np.ndarray]:
    with open(calib_file) as f:
        lines = f.readlines()

    # 使用列表推导式读取数据
    matrices = [
        np.array(lines[i].strip().split(" ")[1:], dtype=np.float32) for i in range(7)
    ]

    return {
        "P0": matrices[0].reshape(3, 4),
        "P1": matrices[1].reshape(3, 4),
        "P2": matrices[2].reshape(3, 4),
        "P3": matrices[3].reshape(3, 4),
        "R0": matrices[4].reshape(3, 3),
        "Tr_velo_to_cam": matrices[5].reshape(3, 4),
        "Tr_imu_to_velo": matrices[6].reshape(3, 4),
    }


class Calibration(object):
    """kitti calibration class

    Args:
        calib (Union[str, dict]): calibration file path or calibration dict

    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.
    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
                = K * [1|t]

    image2 coord:
        ----> x-axis (u)
    |
    |
    v y-axis (v)

    velodyne coord:
    front x, left y, up z

    rect/ref camera coord:
    right x, down y, front z

    Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    """

    def __init__(self, calib: Union[str, dict]) -> None:
        if isinstance(calib, str):
            calib = get_calib_from_file(calib)
        self.P0 = calib["P0"]  # 3 x 4
        self.P1 = calib["P1"]  # 3 x 4
        self.P2 = calib["P2"]  # 3 x 4
        self.R0 = calib["R0"]  # 3 x 3
        self.V2C = calib["Tr_velo_to_cam"]  # 3 x 4
        self.C2V = inverse_rigid_trans(self.V2C)
        self.Tr_imu_to_velo = calib["Tr_imu_to_velo"]  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)  # nx4
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = (
            pts_2d_hom[:, 2] - self.P2.T[3, 2]
        )  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1
        )
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2
        )  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

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
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, "%.8f != %.8f" % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)

        return np.column_stack((x, y, z))

    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
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


def cart_to_hom(pts: np.ndarray) -> np.ndarray:
    """
    将笛卡尔坐标转换为齐次坐标
    :param pts: 形状为 (N, 3 或 2) 的 numpy 数组，表示 N 个点的笛卡尔坐标
    :return pts_hom: 形状为 (N, 4 或 3) 的 numpy 数组，表示 N 个点的齐次坐标
    """
    return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))


def inverse_rigid_trans(Tr: np.ndarray) -> np.ndarray:
    """逆刚体变换矩阵(3x4 形式为 [R|t])
    [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


if __name__ == "__main__":
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    print(cv2.cuda.getCudaEnabledDeviceCount())

    for i in range(1000):
        result = cart_to_hom(np.random.rand(1000, 3))

    profiler.stop()
    profiler.print()

    with open("profiler.html", "w") as f:
        f.write(profiler.output_html())