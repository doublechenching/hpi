#encoding: utf-8
from __future__ import print_function
from keras import backend as K
from config import config as cfg
from training import init_env, get_number_of_steps
gpus = '0, 1'
from proc.data import load_train_csv, split_train_val
from proc.gennerator import BaseGenerator
from matplotlib import pyplot as plt
from skimage.util.montage import montage2d
import numpy as np
import tensorflow as tf

tf.losses.sparse_softmax_cross_entropy

def check_batch_sample(gen, path=None):
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25)
    train_gen = BaseGenerator(train_df, cfg.train_dir, batch_size=cfg.batch_size,
                              aug_args=cfg.aug_args,
                              target_shape=cfg.input_shape[:2],
                              use_yellow=False)
    batch_x, batch_y = next(train_gen)
    x = montage2d(np.squeeze(batch_x[:, :, :, 0]))
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(x, cmap='bone')
    plt.axis('off')
    if path:
        plt.savefig(path)
    else:
        plt.show()



