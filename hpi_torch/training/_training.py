import os
import tqdm
from .callback import Callbacks
from collections import defaultdict
from typing import Type
from torch import optim
import torch
from torch import nn
import numpy as np

optimizers = {
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}

class MetricsCollection:
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}

class Estimator:
    def __init__(self,
                 cfg,
                 model: torch.nn.Module,
                 optimizer: Type[optim.Optimizer],
                 losses,
                 metrics,
                 log_dir
                 ):
        self.model = nn.DataParallel(model, device_ids=cfg.gpu_ids).cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=cfg.lr)
        self.start_epoch = 0
        self.log_dir = log_dir
        self.iter_size = cfg.iter_size
        self.lr_scheduler = None
        self.lr = cfg.lr
        self.cfg = cfg
        self.optimizer_type = optimizer
        self.losses = losses
        self.metrics = metrics


    def resume(self, checkpoint_name, pop_list=[]):
        try:
            checkpoint = torch.load(os.path.join(self.log_dir, checkpoint_name))
        except FileNotFoundError:
            print("resume failed, file not found")
            return False
        self.start_epoch = checkpoint['epoch']
        model_dict = self.model.module.state_dict()
        pretrained_dict = checkpoint['state_dict']
        for item in pop_list:
            print('skip layer', item)
            pretrained_dict.pop(item)
        model_dict.update(pretrained_dict)
        self.model.module.load_state_dict(model_dict)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            pass
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        print("resumed from checkpoint {} on epoch: {}".format(os.path.join(self.log_dir, checkpoint_name), self.start_epoch))

        return True

    def calculate_loss(self, output, target, losses, metrics, training, loss_weights=None):
        n_gpus = len(self.cfg.gpu_ids)
        loss = 0.0
        if not loss_weights:
            loss_weights = {}
            for key in losses.keys():
                loss_weights[key] = 1.0
        loss_values = {}
        for key, loss_func in losses.items():
            value = loss_func(output, target)
            loss = loss + loss_weights[key]*value
            loss_values[key] = value.data.cpu().numpy()[0] / n_gpus
        loss = loss / n_gpus
        loss_values['loss'] = loss.cpu().numpy()[0]

        metrics_values = {}
        for key, metric_func in metrics.items():
            value = metric_func(output, target)
            metrics_values[key] = value.cpu().numpy()[0] / n_gpus

        if training:
            loss.backward()

        return loss_values, metrics_values


    def train_on_batch(self, batch_x, batch_y, training):
        n_gpus = len(self.cfg.gpu_ids)
        if training:
            self.optimizer.zero_grad()
        inputs = batch_x.chunk(n_gpus)
        targets = batch_y.chunk(n_gpus)
        losses_meter = defaultdict(float)
        metrics_meter = defaultdict(float)
        for input, target in zip(inputs, targets):
            input = torch.autograd.Variable(input.cuda(async=True), volatile=not training)
            target = torch.autograd.Variable(target.cuda(async=True), volatile=not training)
            output = self.model(input)

            loss_values, metrics_values = self.calculate_loss(output, target, self.losses, self.metrics, training)
            for key in loss_values.keys():
                losses_meter[key] += loss_values[key]
            for key in metrics_values.keys():
                metrics_meter[key] += metrics_values[key]

        if training:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
            self.optimizer.step()
        return losses_meter, metrics_meter


class BaseTrainer:
    def __init__(self, estimator: Estimator, callbacks=None, state_path=None):
        self.estimator = estimator
        self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        self.metrics_collection = MetricsCollection()
        if state_path:
            self.estimator.resume(state_path)
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def train_one_epoch(self, epoch, loader, training=True):
        avg_meter = defaultdict(float)
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Epoch {}{}".format(epoch, ' eval' if not training else ""), ncols=0)
        for i, data in pbar:
            self.callbacks.on_batch_begin(i)
            losses_meter, metrics_meter = self.estimator.train_on_batch(data['image'], data['mask'], training)
            batch_meter = losses_meter.copy().update(metrics_meter)
            for k, val in batch_meter.items():
                avg_meter[k] += val

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})
            self.callbacks.on_batch_end(i)
        return {k: v / len(loader) for k, v in avg_meter.items()}


    def fit(self, train_loader, val_loader, nb_epoch):
        self.callbacks.on_train_begin()

        for epoch in range(self.estimator.start_epoch, nb_epoch):
            self.callbacks.on_epoch_begin(epoch)
            if self.estimator.lr_scheduler is not None and epoch >= self.estimator.cfg.warmup:
                self.estimator.lr_scheduler.step(epoch)
            # train mode
            self.estimator.model.train()
            self.metrics_collection.train_metrics = self.train_one_epoch(epoch, train_loader, training=True)
            # eval mode
            self.estimator.model.eval()
            self.metrics_collection.val_metrics = self.train_one_epoch(epoch, val_loader, training=False)
            self.callbacks.on_epoch_end(epoch)
            if self.metrics_collection.stop_training:
                break
                
        self.callbacks.on_train_end()