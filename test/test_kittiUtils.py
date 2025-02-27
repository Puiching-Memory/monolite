import sys
sys.path.append('./')
# import pytest,pytest_benchmark

from lib.datasets.kittiUtils import Object3d,get_objects_from_label

test_label_path = r"C:\Users\11386\Downloads\kitti3d\training\label_2\000000.txt"
test_object:Object3d = get_objects_from_label(test_label_path)[0]

def test_get_obj_level(benchmark):
    result = benchmark(test_object.get_obj_level)

def test_generate_corners3d(benchmark):
    result = benchmark(test_object.generate_corners3d)
    
def test_generate_label_matrix(benchmark):
    result = benchmark(test_object.generate_label_matrix)