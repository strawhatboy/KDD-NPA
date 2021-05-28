import colorlog
import logging
import torch
import numpy as np

logger = logging.getLogger('util')

def init_logging():
    colorlog.basicConfig(
        level=logging.INFO,
        format='%(log_color)s%(asctime)s - %(name)s [%(levelname)s] %(white)s%(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def get_device(device_id=0):
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    logger.info('Got device: {}'.format(device))
    return device

def set_seeds(s):
    logger.info('Seed set to {}'.format(s))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    np.random.seed(s)

    #???
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


