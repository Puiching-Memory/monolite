import sys
import os
sys.path.append(os.path.abspath("./"))
from lib.datasets.kitti import KITTI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class data_cfg():
    def __init__(self):
        self.dataset = KITTI
        self.root_dir = r"C:\Users\11386\Downloads\kitti3d"
        self.drop_last_val = True
        self.split = 'trainval'
        self.batch_size = 16 # We used the BN layer, so a value of >=2 is recommended
        self.num_workers = 4
        
        self.use_3d_center = True
        self.writelist = ['Car','Pedestrian','Cyclist']
        self.class_merging = False
        self.use_dontcare = False
        self.data_dir = r"C:\Users\11386\Downloads\kitti3d"
        self.random_flip = 0.5
        self.random_crop = 0.5
        self.scale = 0.4
        self.shift = 0.1
        self.image_size = [384, 1280]

class data_set():
    def __init__(self, cfg:dict):
        dataset = cfg['dataset']
        self.train_set = dataset(root_dir=cfg['root_dir'],cfg=cfg)
        self.train_loader = DataLoader(dataset=self.train_set,
                                        batch_size=cfg['batch_size'],
                                        num_workers=cfg['num_workers'],
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True,
                                        persistent_workers=True,
                                        prefetch_factor=2)
        
        self.val_set = dataset(root_dir=cfg['root_dir'],cfg=cfg)
        self.val_loader = DataLoader(dataset=self.val_set,
                                batch_size=cfg['batch_size'],
                                num_workers=cfg['num_workers'],
                                shuffle=False,
                                pin_memory=True,
                                drop_last=cfg['drop_last_val'])
        
        self.test_set = dataset(root_dir=cfg['root_dir'],cfg=cfg)
        self.test_loader = DataLoader(dataset=self.test_set,
                                batch_size=1,
                                num_workers=1,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=False)
        
if __name__ == '__main__':
    cfg = data_cfg()
    data = data_set(vars(cfg))
    for i, (inputs, coord_range, targets, info) in enumerate(data.train_loader):
        print(inputs.shape)
        print(coord_range.shape)
        print(targets)
        print(info)
        break