#encoding: utf-8
from __future__ import print_function
from keras import backend as K
from config import config as cfg
from training import init_env, get_number_of_steps
gpus = '0, 1'
init_env('gpus')
n_gpus = len(gpus.split(','))
import os
from proc.data import load_train_csv, split_train_val
from proc.gennerator import BaseGenerator
from model import Xception
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from training.callback import MultiGPUCheckpoint
from utils import makedir
from keras.utils import multi_gpu_model
import tensorflow as tf
from metrics import roc_auc_loss
from keras.optimizers import Adam


def train():
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25)
    train_gen = BaseGenerator(train_df, cfg.train_dir, batch_size=cfg.batch_size,
                              aug_args=cfg.aug_args,
                              target_shape=cfg.input_shape[:2],
                              use_yellow=False)

    val_gen = BaseGenerator(val_df, cfg.train_dir, batch_size=cfg.batch_size,
                            aug_args=cfg.aug_args,
                            target_shape=(512, 512),
                            use_yellow=False)
    if n_gpus > 0:
        with tf.device('/cpu:0'):
            cpu_model = Xception(cfg.input_shape, include_top=True, n_class=len(cfg.label_names))
            model = multi_gpu_model(cpu_model, gpus=n_gpus)
    else:
        model = Xception(cfg.input_shape, include_top=True, n_class=len(cfg.label_names))

    model.compile(optimizer=Adam(1e-3), loss=roc_auc_loss,
                  metrics=['binary_accuracy', 'mae'])
    log_dir = os.path.join(cfg.log_dir, 'base_xception')
    makedir(log_dir)
    weights_path = os.path.join(log_dir, cfg.weights_file)

    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 mode='min', save_weights_only=True)
    
    if n_gpus > 0:
        del checkpoint
        checkpoint = MultiGPUCheckpoint(weights_path, cpu_model, monitor='val_loss')
    
    callbacks = [checkpoint]
    callbacks += [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')]
    train_steps = get_number_of_steps(len(train_df), cfg.batch_size)
    val_steps   = get_number_of_steps(len(val_df), cfg.batch_size)
    model.fit_generator(train_gen, 
                        epochs=cfg.epochs,
                        steps_per_epoch=train_steps,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        workers=cfg.n_works,
                        max_queue_size=cfg.n_queue,
                        use_multiprocessing=True,
                        validation_steps=val_steps,
                        initial_epoch=0)


    K.clear_session()


if __name__ == "__main__":
    print(cfg)
    train()
