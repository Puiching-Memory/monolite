import os
import tqdm

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

import torch
import numpy as np
import random
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss,compute_line3d_loss
import pickle
import time
import swanlab
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# torch.autograd.set_detect_anomaly(True) # 梯度异常追踪


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 train_lane_loader,
                 train_samper,
                 train_lane_samper,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 device,
                 **kwargs):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_samper = train_samper
        self.train_lane_samper = train_lane_samper
        self.train_lane_loader = train_lane_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.point_range = torch.tensor([[90,6,3]]).T
        self.anchor = build_anchor([5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,60,75,90], [6,2,-2,-6]).T /self.point_range
        self.anchor = self.anchor.cuda()
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda') # > torch=2.0.0
        #self.scaler =  torch.cuda.amp.GradScaler() # < torch=2.0.0
        
        if local_rank == 0:
            swanlab.init(
                project=f"Monodle--{time.time()}",
                config=self.cfg,
                #mode='local',
                logdir='./log'
            )
            self.writer = SummaryWriter('./log/')
            

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        #self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(self.device)



    def train(self):
        start_epoch = self.epoch
        if local_rank == 0:
            progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_lane_samper.set_epoch(random.randint(0,999))
            self.train_samper.set_epoch(random.randint(0,999))
            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
                
            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            if local_rank == 0:
                progress_bar.update()

        if local_rank == 0:
            progress_bar.close()
            self.writer.flush()
            self.writer.close()
                
        return None


    def train_one_epoch(self):
        self.model.train()

        train_loader_max = len(self.train_loader)
        train_lane_loader_max = len(self.train_lane_loader)
        train_loader = iter(self.train_loader)
        train_lane_loader = iter(self.train_lane_loader)

        if local_rank == 0:
            progress_bar = tqdm.tqdm(total=min(train_loader_max,train_lane_loader_max), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')

        train_range = range(min(train_loader_max,train_lane_loader_max))
        for batch_idx in train_range:
            total_loss1 = torch.tensor(0.,device=self.device,requires_grad=True)
            total_loss2 = torch.tensor(0.,device=self.device,requires_grad=True)
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
            #with torch.cuda.amp.autocast():
                if batch_idx < train_loader_max:
                    inputs, targets, info = next(train_loader)
                    inputs = inputs.to(self.device,memory_format=torch.channels_last)
                    for key in targets.keys():
                        targets[key] = targets[key].to(self.device)

                    outputs = self.model(inputs)
                    total_loss1, stats_batch1 = compute_centernet3d_loss(outputs, targets)

            self.scaler.scale(total_loss1).backward(retain_graph=True)

            with torch.amp.autocast('cuda'):
            #with torch.cuda.amp.autocast():
                if batch_idx < train_lane_loader_max:
                    inputs2, targets2, info2 = next(train_lane_loader)
                    inputs2 = inputs2.to(self.device,memory_format=torch.channels_last)
                    for key in targets2.keys():
                        targets2[key] = targets2[key].to(self.device)
                    
                    outputs2 = self.model(inputs2)
                    total_loss2, stats_batch2 = compute_line3d_loss(outputs2, targets2,self.anchor)
                    total_loss2 = total_loss2.to(device=self.device)

            self.scaler.scale(total_loss2).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if local_rank == 0:
                ploss = [self.epoch,t2f(info),total_loss1.item(),t2f(stats_batch1)]
                with open(f'./checkpoints/{self.epoch}_train_info.pkl','wb') as pfile:
                    pickle.dump(ploss,pfile)
                swanlab.log({'epoch':self.epoch,
                             'loss1':total_loss1,
                             'loss2':total_loss2,
                             **stats_batch1,
                             'get_item_time1':torch.mean(info['get_item_time']).float(),
                             'get_item_time2':torch.mean(info2['get_item_time']).float()
                             })
            
                # self.writer.add_scalar("Loss/loss1", total_loss1, self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/loss2", total_loss2, self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/seg", stats_batch1['seg'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/offset2d", stats_batch1['offset2d'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/size2d", stats_batch1['size2d'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/offset3d", stats_batch1['offset3d'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/detph", stats_batch1['depth'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/size3d", stats_batch1['size3d'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/heading", stats_batch1['heading'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/loss_time", stats_batch1['losstime'], self.epoch*batch_idx)
                # self.writer.add_scalar("Loss/data_time", torch.mean(info['get_item_time']).float(), self.epoch*batch_idx)
                progress_bar.update()

        if local_rank == 0:
            #gen_model_map(outputs2,targets2)
            pass




def t2f(input_dict):  
    """  
    将字典中所有可能的tensor类型转换为Python的float。  
    这里我们假设对于每个tensor，我们只取其第一个元素作为float值。  

    参数:  
    - input_dict: 包含可能tensor的字典。  
      
    返回:  
    - 转换后的字典，其中tensor被替换为它们第一个元素的float值。  
    """  
    output_dict = {}  
    for key, value in input_dict.items():  
        if isinstance(value, torch.Tensor):  
            # 检查tensor是否为空（即没有元素）  
            if value.numel() == 1:  
                # 取tensor的第一个元素并转换为float  
                output_dict[key] = float(value.item())  # 对于单个元素的tensor，使用.item()  
            elif value.numel() > 1:
                output_dict[key] = value.tolist()
            else:  
                # 如果tensor为空，可以选择不添加这个键，或者设置一个默认值  
                output_dict[key] = None  # 或者 float('nan')，根据你的需求  
        else:  
            # 如果不是tensor，则直接保留原值  
            output_dict[key] = value  
    return output_dict  

def gen_model_map(output,target):
    anchors = build_anchor([10,20,40,60,80,100], [2,1,-1,-2]).T / 120

    # print(output['line_reg'].shape,target['distance'].shape)
    ouptput_p = output['line_reg'][0].reshape(30,3).cpu().detach().T      # 模型输出偏置
    target_p = target['distance'][0].cpu().detach().T                  # 真正的偏置
    
    target_line = target_p + anchors
    ouptput_line = ouptput_p + anchors
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  
    ax.scatter(target_line[0], target_line[1], target_line[2],c=target_line[2], cmap='viridis',alpha=0.5)  # c=z 根据z值的不同来设置颜色 
    ax.scatter(anchors[0],anchors[1],anchors[2],c='black',marker='x') # 原始锚点
    ax.scatter(ouptput_line[0],ouptput_line[1],ouptput_line[2],c='red',marker='x') # 模型偏移后锚点
    #ax.scatter(target_p[0],target_p[1],target_p[2],c='red',marker='x')
    ax.set_xlabel('X Label')  
    ax.set_ylabel('Y Label')  
    ax.set_zlabel('Z Label')  
    # 显示图形
    # plt.savefig(r'/desay/file_warehouse/ids/upload/zk/monodle/point.jpg')
    swanlab.log({"plot": swanlab.Image(plt)})

def build_anchor(anchor_x,anchor_y,anchor_z=-2):
    points = []
    for x in anchor_x:
        for y in anchor_y:
            points.append((x,y,anchor_z))
    
    points = torch.tensor(points)
    return points