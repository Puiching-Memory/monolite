import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torchinfo import summary

#import optuna

parser = argparse.ArgumentParser(description='End-to-End Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('--ddp', action='store_true',default=False, help='ddp')

args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = create_logger(log_file)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
        world_size = os.environ['WORLD_SIZE']

    if distributed:
        dist.init_process_group(backend="nccl", init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        train_loader, test_loader,train_lane_loader,train_samper,test_samper,train_lane_samper  = build_dataloader(cfg['dataset'],world_size,rank,cfg['dataset']['workers'])
        model = build_model(cfg['model']).to(device)
        #device = f"cuda:{local_rank}"
        # model = torch.compile(model,mode="reduce-overhead")
    else:
        train_loader, test_loader,train_lane_loader,train_samper,test_samper,train_lane_samper  = build_dataloader(cfg['dataset'],-1,-1,cfg['dataset']['workers'])
        model = build_model(cfg['model']).cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model).to(device=device)
        
    if local_rank == 0:
        logger.info(f"\n{summary(model, input_size=(cfg['dataset']['batch_size'], 3, 384, 1280), depth=3, verbose=0)}")

    if args.evaluate_only:
        if local_rank == 0:
            logger.info('###################  Evaluation Only  ##################')

        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        device=device)
        tester.test()
        return  

    if distributed:
        logger.debug(f'>> local_rank: {local_rank} {rank}')
        # NOTE:https://pytorch.ac.cn/tutorials/intermediate/memory_format_tutorial.html
        model = model.to(memory_format=torch.channels_last) # memory swich format
        #model = torch.compile(model,mode='max-autotune')
        #model = torch.compile(model)
        model = DDP(model, device_ids=[device],broadcast_buffers=True, find_unused_parameters=True,gradient_as_bucket_view=False)
        #model = FSDP(model,sync_module_states=True,device_id=local_rank)
        

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    if local_rank == 0:
        logger.info('###################  Training  ##################')
        logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
        logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
        logger.info(f"Optimizer:{cfg['optimizer']['type']}")
    
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      train_lane_loader=train_lane_loader,
                      train_samper=train_samper,
                      train_lane_samper=train_lane_samper,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      device=device)
    trainer.train()

    if distributed:
        if local_rank == 0:
            logger.warning("skip Evaluation in DDP model")
        return
    
    logger.info('###################  Evaluation  ##################' )
    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger,
                    device=device)
    tester.test()

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()