import torch
import numpy as np
from torch.utils.data import DataLoader,DistributedSampler,Sampler,ConcatDataset
#from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_dataset_next import KITTI_Dataset
from lib.datasets.openlane import OpenLane_dataset
import os
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg,world_size,rank,workers=4):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split='train', cfg=cfg)
        test_set = KITTI_Dataset(split='val', cfg=cfg)
    if cfg['type'] == 'KITTI&OpenLane':
        train_set = KITTI_Dataset(split='train', cfg=cfg)
        test_set = KITTI_Dataset(split='val', cfg=cfg)
        train_lane_set = OpenLane_dataset(r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/images',
                                          r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/lane3d_1000/training',
                                          [90,6,3])
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    if local_rank != -1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
        train_lane_sampler = DistributedSampler(train_lane_set, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        test_sampler = None
        train_lane_sampler = None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              persistent_workers=True,
                              prefetch_factor=8)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False,
                             sampler=test_sampler,
                             prefetch_factor=8)
    
    train_lane_loader = DataLoader(dataset=train_lane_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_lane_sampler,
                              persistent_workers=True,
                              prefetch_factor=8)

    return train_loader, test_loader,train_lane_loader,train_sampler,test_sampler,train_lane_sampler
