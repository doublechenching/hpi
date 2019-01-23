#encoding: utf-8
from __future__ import print_function
from keras import backend as K
from config import config as cfg
from training import init_env, get_number_of_steps
gpus = '1'
init_env(gpus)
n_gpus = len(gpus.split(','))
cfg.batch_size = cfg.batch_size * n_gpus
import os
from proc.data import load_train_csv, split_train_val
from proc.gennerator import BaseGenerator
from model.resnet import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from training.callback import MultiGPUCheckpoint
from utils import makedir
from keras import optimizers as KO
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
from proc.gennerator import BaseTestGenerator
import pandas as pd
from metrics import f1_score
from predict_resnet import predict_on_gennerator


def lr_schedule(epoch, lr=1e-2):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 30, 60, 90, 120 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    if epoch > 50:
        if epoch % 10 == 9:
            lr *= 0.5e-3
        elif 6 <= epoch % 10 < 9:
            lr *= 1e-3
        elif 3 <= epoch % 10 < 6:
            lr *= 1e-2
        elif epoch % 10 < 3:
            lr *= 1e-1

    print('Learning rate: ', lr)

    return lr


def load_train_gennerator(batch_size):
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)
    train_gen = BaseTestGenerator(train_df, cfg.train_dir,
                                  batch_size=batch_size,
                                  aug_args=cfg.aug_args,
                                  target_shape=cfg.input_shape[:2],
                                  use_yellow=False,
                                  return_label=True,
                                  preprocessing_function=preprocess_input
                                  )
    return train_gen, train_df


def load_val_gennerator(batch_size):
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)
    val_gen = BaseTestGenerator(val_df, cfg.train_dir,
                                batch_size=batch_size,
                                aug_args=cfg.aug_args,
                                target_shape=cfg.input_shape[:2],
                                use_yellow=False,
                                return_label=True,
                                preprocessing_function=preprocess_input
                                )
    return val_gen, val_df


def pretrain(task_name='base_resnet',
             epochs=10,
             lr=1e-1,
             target_shape=(512, 512),
             trainable=True,
             pretrain_weights='imagenet',
             init_epoch=0):
    cfg.input_shape = list(target_shape) + [3]
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)
    train_gen = BaseGenerator(train_df, cfg.train_dir, batch_size=cfg.batch_size,
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
            cpu_model = ResNet50(input_shape=cfg.input_shape, classes=len(
                cfg.label_names), trainable=trainable, weights=pretrain_weights)
        model = multi_gpu_model(cpu_model, gpus=n_gpus)
    else:
        print('use single gpu')
        model = ResNet50(input_shape=cfg.input_shape, classes=len(
            cfg.label_names), trainable=trainable, weights=pretrain_weights)
    model.compile(optimizer=KO.Adam(lr=lr, amsgrad=True), loss='binary_crossentropy',
                  metrics=[f1_score, 'mae'])
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    weights_path = os.path.join(log_dir, cfg.weights_file)
    checkpoint = ModelCheckpoint(weights_path, monitor='f1_score',
                                 verbose=1, save_best_only=True,
                                 mode='max', save_weights_only=True)
    if n_gpus > 1:
        del checkpoint
        checkpoint = MultiGPUCheckpoint(weights_path, cpu_model, verbose=1,
                                        monitor='f1_score',
                                        mode='max',
                                        save_weights_only=True,
                                        save_best_only=True)
    callbacks = [checkpoint]
    callbacks += [ReduceLROnPlateau(monitor='f1_score',
                                    factor=0.5, patience=3, verbose=1, mode='max')]
    # callbacks += [LearningRateScheduler(lr_schedule)]
    train_steps = get_number_of_steps(len(train_df), cfg.batch_size)
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
                        initial_epoch=init_epoch)
    K.clear_session()


def select_hard_samples(test_gen, df, pretrain_weights):
    model = ResNet50(input_shape=cfg.input_shape,
                     classes=len(cfg.label_names),
                     trainable=True,
                     weights=pretrain_weights)
    select_idx = []
    for batch_id in range(len(test_gen)):
        print('processing ', batch_id, ' th batch, total ', len(test_gen))
        batch_x, batch_y = test_gen[batch_id]
        batch_df_idx = test_gen.batch_indexes
        batch_pred = model.predict(batch_x, batch_size=len(batch_x))
        batch_pred = np.split(batch_pred, int(len(batch_x)/8), axis=0)
        for i, pred in enumerate(batch_pred):
            pred = np.mean(pred, axis=0, keepdims=True)
            if not np.alltrue(np.round(pred[:]) == batch_y[i, :]):
                select_idx.append(batch_df_idx[i])
                print('select hard sample', df.iloc[batch_df_idx[i], 0])
    
    return df.iloc[select_idx]


def select_hard_samples_mp(test_gen, df, pretrain_weights):
    pred_Y, test_Y = predict_on_gennerator(
        test_gen, pretrain_weights, return_label=True)
    select_idx = []
    for idx, (pred, gt) in enumerate(zip(pred_Y, test_Y)):
        if np.alltrue(np.round(pred) == gt):
            select_idx.append(idx)
            print('select hard sample', df.iloc[idx, 0])
    return df.iloc[select_idx]


def train_on_hard(task_name, pretrain_weights, target_shape=(512, 512),
                  trainable=True, lr=1e-2, epochs=3, use_multiprocessing=True):
    batch_size = int(cfg.batch_size / n_gpus / 8)
    print('test batch size', batch_size)
    train_gen, train_df = load_train_gennerator(batch_size)
    
    if use_multiprocessing:
        hard_df = select_hard_samples_mp(train_gen, train_df, weights)
    else:
        hard_df = select_hard_samples(train_gen, train_df, weights)
    
    cfg.input_shape = list(target_shape) + [3]
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)
    hard_gen = BaseGenerator(hard_df, cfg.train_dir, batch_size=cfg.batch_size,
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
            cpu_model = ResNet50(input_shape=cfg.input_shape, classes=len(
                cfg.label_names), trainable=trainable, weights=pretrain_weights)
        model = multi_gpu_model(cpu_model, gpus=n_gpus)
    else:
        print('use single gpu')
        model = ResNet50(input_shape=cfg.input_shape, classes=len(
            cfg.label_names), trainable=trainable, weights=pretrain_weights)
    model.compile(optimizer=KO.Adam(lr=lr, amsgrad=True), loss='binary_crossentropy',
                  metrics=[f1_score, 'mae'])
    log_dir = os.path.join(cfg.log_dir, task_name)
    makedir(log_dir)
    weights_path = os.path.join(log_dir, cfg.weights_file)
    checkpoint = ModelCheckpoint(weights_path, monitor='f1_score',
                                 verbose=1, save_best_only=True,
                                 mode='max', save_weights_only=True)
    if n_gpus > 1:
        del checkpoint
        checkpoint = MultiGPUCheckpoint(
            weights_path, cpu_model, monitor='f1_score', save_weights_only=True, save_best_only=True, mode='max')
    callbacks = [checkpoint]
    callbacks += [ReduceLROnPlateau(monitor='f1_score',
                                    factor=0.5, patience=3, verbose=1, mode='max')]
    train_steps = get_number_of_steps(len(hard_df), cfg.batch_size)
    val_steps = get_number_of_steps(len(val_df), cfg.batch_size)
    model.fit_generator(hard_gen,
                        epochs=epochs,
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
    # pretrain 1
    # task_name = 'pretrain_resnet10'
    # epochs = 10
    # pretrain(task_name=task_name, epochs=epochs, lr=1e-1, target_shape=(256, 256), pretrain_weights='imagenet', trainable=False)
    # # pretrain 2
    # pretrain_weights = os.path.join(cfg.log_dir, task_name, cfg.weights_file).format(epoch=epochs)
    # task_name = 'pretrain_resnet50'
    # epochs = 30
    # pretrain(task_name=task_name, epochs=epochs, lr=1e-2, target_shape=(384, 384), pretrain_weights=pretrain_weights, trainable=True)
    # pretrain 3
    task_name = 'train_resnet70'
    # pretrain_weights = os.path.join(cfg.log_dir, task_name, cfg.weights_file).format(epoch=50)
    # epochs = 100
    # pretrain(task_name=task_name, epochs=epochs, lr=1e-3, target_shape=(512, 512),
    #          pretrain_weights=pretrain_weights, trainable=True, init_epoch=50)
    weights = os.path.join(cfg.log_dir, task_name,
                           cfg.weights_file).format(epoch=99)
    for i in range(30):
        task_name = 'toh'+str(i)
        train_on_hard(task_name, weights, target_shape=(512, 512), lr=1e-4, epochs=3)
        weights = os.path.join(cfg.log_dir, task_name, cfg.weights_file).format(epoch=3)
