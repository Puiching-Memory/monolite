import sys
sys.path.append('./')
# import pytest,pytest_benchmark
import numpy as np

from lib.datasets.kittiUtils import Object3d,get_objects_from_label,Calibration,get_calib_from_file

# test_label
test_label_path = r"C:\Users\11386\Downloads\kitti3d\training\label_2\000000.txt"
test_object:Object3d = get_objects_from_label(test_label_path)[0]

def test_get_obj_level(benchmark):
    result = benchmark(test_object.get_obj_level)

def test_generate_corners3d(benchmark):
    result = benchmark(test_object.generate_corners3d)
    
def test_generate_label_matrix(benchmark):
    result = benchmark(test_object.generate_label_matrix)
    
# test_calib
test_calib_file = r"C:\Users\11386\Downloads\kitti3d\training\calib\000000.txt"
test_calib:Calibration = get_calib_from_file(test_calib_file)

def test_lidar_to_rect(benchmark):
    result = benchmark(test_calib.lidar_to_rect, np.random.randn(1,3))
    
def test_rect_to_lidar(benchmark):
    result = benchmark(test_calib.rect_to_lidar, np.random.randn(1,3))
    
def test_rect_to_img(benchmark):
    result = benchmark(test_calib.rect_to_img, np.random.randn(1,3))
    
def test_lidar_to_img(benchmark):
    result = benchmark(test_calib.lidar_to_img, np.random.randn(1,3))

def test_img_to_rect(benchmark):
    result = benchmark(test_calib.img_to_rect, np.random.randint(0,9999),np.random.randint(0,9999),np.random.rand(1))
    
def test_depthmap_to_rect(benchmark):
    result = benchmark(test_calib.depthmap_to_rect, np.random.randn(2000,2000))
    
def test_corners3d_to_img_boxes(benchmark):
    result = benchmark(test_calib.corners3d_to_img_boxes, np.random.randn(1,8,3))
    
def test_camera_dis_to_rect(benchmark):
    result = benchmark(test_calib.camera_dis_to_rect, np.random.randint(0,9999),np.random.randint(0,9999),np.random.randint(0,9999))
    
def test_alpha2ry(benchmark):
    result = benchmark(test_calib.alpha2ry, np.random.randint(-314,314)/100,np.random.randint(0,9999))
    
def test_ry2alpha(benchmark):
    result = benchmark(test_calib.ry2alpha, np.random.randint(-314,314)/100,np.random.randint(0,9999))