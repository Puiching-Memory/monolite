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
import torch.distributed as dist

import optuna

parser = argparse.ArgumentParser(description='End-to-End Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('--ddp', action='store_true',default=False, help='ddp')

args = parser.parse_args()

def objective(trail):
    batch_size = trail.suggest_int('batchsize', 1, 16)
    lr = trail.suggest_float('lr', 1e-4, 1e-2,step=0.0001)
    #lossfunc = trail.suggest_categorical('loss', ['MSE', 'MAE'])
    #opt = trail.suggest_categorical('opt', ['adam', 'adamw','sgd','adai'])
    #hidden_layer = trail.suggest_int('hiddenlayer', 20, 1200)
    #activefunc = trail.suggest_categorical('active', ['relu', 'sigmoid', 'tanh'])
    weight_decay = trail.suggest_float('weight_dekay', 0, 1,step=0.01)
    #momentum= trail.suggest_float('momentum',0,1,step=0.01)

    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = create_logger(log_file)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 覆写cfg
    cfg['dataset']['batch_size'] = batch_size
    cfg['optimizer']['lr'] = lr
    cfg['optimizer']['weight_decay'] = weight_decay
    logger.debug(f'覆写cfg:batch_size{batch_size}/lr{lr}/weight_decay{weight_decay}')

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
        world_size = os.environ['WORLD_SIZE']
        logger.debug(f'distributed{distributed}')
        logger.debug(f'world_size{world_size}')

    if distributed:
        dist.init_process_group(backend="nccl", init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        logger.debug(f'world_size{world_size} rank{rank}')
        torch.cuda.set_device(local_rank)
        train_loader, test_loader  = build_dataloader(cfg['dataset'],world_size,rank,cfg['dataset']['workers'])
        model = build_model(cfg['model']).cuda()
        device = f"cuda:{local_rank}"
        model = model.to(device=device)
    else:
        train_loader, test_loader  = build_dataloader(cfg['dataset'],-1,-1,cfg['dataset']['workers'])
        model = build_model(cfg['model']).cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model).to(device=device)

    if distributed:
        logger.debug(f'>> local_rank: {local_rank}')
        model = DDP(model, broadcast_buffers=True, find_unused_parameters=True,gradient_as_bucket_view=True)

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
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      device=device)
    trainer.train()


def main():
    study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study.
    study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.

if __name__ == '__main__':
    main()