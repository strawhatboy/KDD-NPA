
import pretty_errors
import argparse
from config import train_config
from util import init_logging, set_seeds, get_device
from dataset import get_dataloader
from train import train
from model import NPAModel
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger('train_arg')


def init_argparse():
    parser = argparse.ArgumentParser(description='Trainer for hw3')
    parser.add_argument('--runid', default='default_run', type=str, help='the runid used for names of model, result, plot files')
    parser.add_argument('--epochs', type=int, default=3, help='epochs that will be trained')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--model_savepath', type=str, default='./model.mdl', help='model saved path under \'model\' folder')
    # parser.add_argument('--train_dir', type=str, default='./training/labeled', help='training data path')
    # parser.add_argument('--val_dir', type=str, default='./validation', help='validation data path, set to empty for final run')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay for optimizer')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    # parser.add_argument('--semi_threshold', type=float, default=0.65, help='semi-supervising learning threshold')
    

    return parser.parse_args()
    

def hack_config(config, hacked):
    config['runid'] = hacked['runid']
    config['epochs'] = hacked['epochs']
    config['batch_size'] = hacked['batch_size']
    config['optimizer_params']['lr'] = hacked['lr']
    # config['optimizer_params']['weight_decay'] = hacked['weight_decay']
    config['model_savepath'] = './{}.mdl'.format(hacked['runid'])
    return config

if __name__ == '__main__':
    init_logging()
    set_seeds(0)
    args = vars(init_argparse())
    hack_config(train_config, args)
    device = get_device(args['gpu'])
    train_config['device'] = device
    logger.info('Going to train with config: {}'.format(train_config))

    # load data
    train_loader, val_loader, user_len, word_len, embedding_mat = get_dataloader(train_config['batch_size'])

    summary_writer = SummaryWriter('tensorlog/{}'.format(train_config['runid']), 'hahahahahaha')

    model = NPAModel(user_len, word_len, embedding_mat).to(device)
    train(model, train_loader, val_loader, summary_writer, config=train_config)

    



