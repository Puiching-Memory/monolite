import torch
import numpy as np
#import logging
import random
from loguru import logger
import os
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

def create_logger(log_file, rank=0):
    # log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    # logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
    #                     format=log_format,
    #                     filename=log_file)
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    # console.setFormatter(logging.Formatter(log_format))
    # logging.getLogger(__name__).addHandler(console)
    # return logging.getLogger(__name__)
    log_path = './log'
    logger.add( f"{log_path}/{log_file}.log",backtrace=True,enqueue=True)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True