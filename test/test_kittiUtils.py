import sys
sys.path.append('./')
# import pytest,pytest_benchmark

from lib.datasets.kittiUtils import *


def test_get_objects_from_label(benchmark):
    label_path = r"C:\Users\11386\Downloads\kitti3d\training\label_2\000000.txt"
    # objects = get_objects_from_label(label_path)
    result = benchmark(get_objects_from_label, label_path)
    