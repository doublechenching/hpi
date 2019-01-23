# encoding: utf-8
import random
from utils import schedule_steps, find_best_weights
from metrics import f1_score
from keras import optimizers as KO
import tensorflow as tf
from keras.utils import multi_gpu_model
from utils import makedir
from training.callback import MultiGPUCheckpoint
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from model._xception import Xception, preprocess_input
from proc.gennerator import BaseGenerator
from proc.data import load_train_csv, split_train_val
import numpy as np
import os
from __future__ import print_function
from keras import backend as K
from config import config as cfg
from training import init_env, get_number_of_steps
gpus = '3'
init_env(gpus)
n_gpus = len(gpus.split(','))


def train(task_name='base_xception',
          epochs=6,
          target_shape=(512, 512),
          lr_schedule=None,
          weights='imagenet',
          trainable=True,
          seed=42,
          save_best_only=True,
          initial_epoch=0,
          drop_rate=0):
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)
    np.random.seed(seed + 111)
    random.seed(seed + 111)
    tf.set_random_seed(seed + 111)

    train_gen = BaseGenerator(train_df, cfg.train_dir,
                              batch_size=cfg.batch_size,
                              aug_args=cfg.aug_args.copy(),
                              target_shape=target_shape,
                              use_yellow=False,
                              preprocessing_function=preprocess_input)
    val_gen = BaseGenerator(val_df, cfg.train_dir, batch_size=cfg.batch_size,
                            aug_args=cfg.aug_args.copy(),
                            target_shape=target_shape,
                            use_yellow=False,
                            preprocessing_function=preprocess_input)

    if n_gpus > 1:
        print('use multi gpu')
        with tf.device('/cpu:0'):
            cpu_model = Xception(cfg.input_shape,
                                 include_top=True,
                                 n_class=len(cfg.label_names),
                                 weights=weights,
                                 trainable=trainable,
                                 drop_rate=drop_rate)
        model = multi_gpu_model(cpu_model, gpus=n_gpus)
    else:
        print('use single gpu')
        model = Xception(cfg.input_shape,
                         include_top=True,
                         n_class=len(cfg.label_names),
                         weights=weights,
                         trainable=trainable,
                         drop_rate=drop_rate)

    model.compile(optimizer=KO.Adam(lr=lr_schedule[0][0], amsgrad=True),
                  loss='binary_crossentropy',
                  metrics=[f1_score, 'mae'])
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    weights_path = os.path.join(log_dir, cfg.weights_file)
    checkpoint = ModelCheckpoint(weights_path,
                                 monitor='f1_score',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=save_best_only,
                                 save_weights_only=True)
    if n_gpus > 1:
        del checkpoint
        checkpoint = MultiGPUCheckpoint(weights_path, cpu_model,
                                        monitor='f1_score',
                                        mode='max',
                                        save_best_only=save_best_only)
    callbacks = []
    callbacks += [checkpoint]
    callbacks += [LearningRateScheduler(
        lambda epoch: schedule_steps(epoch, lr_schedule))]
    train_steps = get_number_of_steps(len(train_df), cfg.batch_size) * 4
    val_steps = get_number_of_steps(len(val_df), cfg.batch_size)
    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=train_steps,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        workers=cfg.n_works,
                        max_queue_size=cfg.n_queue,
                        use_multiprocessing=True,
                        validation_steps=val_steps,
                        initial_epoch=initial_epoch)

    del model
    del checkpoint
    K.clear_session()


if __name__ == "__main__":
    print(cfg)
    # pretrain， warm up with lower learning rate
    # improve, use gradual warmup, avoid a sudden increase of the lr,
    # reach peak after 5 epochs
    task_name = 'pretrain_xception3'
    train(task_name=task_name,
          epochs=6,
          lr_schedule=[(1e-5, 2), (3e-3, 4), (1e-3, 6)],
          weights='xception_imagenet.hdf5',
          seed=41,
          trainable=False,
          save_best_only=False)
    # train
    weights = find_best_weights(os.path.join(cfg.log_dir, task_name))
    task_name = 'xception3'
    train(task_name=task_name, epochs=70,
          lr_schedule=[(5e-5, 2), (3e-3, 10), (1e-3, 40),
                       (5e-4, 55), (1e-4, 65), (1e-5, 70)],
          weights=weights, seed=42)
    weights = find_best_weights(os.path.join(cfg.log_dir, task_name))
    train(task_name=task_name,
          epochs=100,
          lr_schedule=[(1e-5, 72), (3e-4, 80), (1e-4, 90), (1e-5, 100)],
          weights=weights, seed=43, initial_epoch=70)
    weights = find_best_weights(os.path.join(cfg.log_dir, task_name))
    # fintune
    task_name = 'xception_tune3'
    train(task_name=task_name,
          epochs=25,
          lr_schedule=[(5e-5, 0), (5e-5, 2), (1e-4, 10), (1e-5, 20), (1e-6, 25)],
          weights=weights,
          seed=44,
          initial_epoch=70)
