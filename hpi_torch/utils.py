from __future__ import print_function
import os
import sys
import torch
import shutil
import numpy as np
from config import config as cfg
import random
import warnings


def save_checkpoint(state, task_name, is_best_loss, is_best_f1):
    epoch = state['epoch']
    filename = os.path.join(cfg.log_dir, task_name, cfg.weights_file).format(epoch=epoch)
    torch.save(state, filename)
    if is_best_loss:
        best_loss_weights = os.path.join(cfg.log_dir, task_name, cfg.best_val_loss_weights)
        shutil.copyfile(filename, best_loss_weights)
        print('saving best val loss model---', best_loss_weights)
    if is_best_f1:
        best_f1_weights = os.path.join(cfg.log_dir, task_name, cfg.best_f1_loss_weights)
        print('saving best f1 loss model---', best_f1_weights)
        shutil.copyfile(filename, best_f1_weights)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger():
    """
    重定向sys.stdout输出流到文件
    """
    def __init__(self, file="print_log.txt", mode="wt"):
        self.file = open(file, mode)
        self.terminal = sys.stdout

    def write(self, line):
        self.file.write(line)
        print(line, end='')

    def flush(self):
        pass

    def __call__(self, line):
        self.write(line)

    def __del__(self):
        self.file.close()


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    lr = lr[0]

    return lr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist folder---', path)


if __name__ == "__main__":
    log = Logger()
    print("hello", "qinchenchen")
    print("hello" , "qinchenchen")
    print("hello", "qinchenchen", file=log)
    print("hello" , "qinchenchen", file=log)