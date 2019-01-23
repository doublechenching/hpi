# encoding: utf-8
import os
from utils import init_env
import numpy as np
import time
from tqdm import tqdm
from config import config as cfg
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from data import get_dataloader
import pandas as pd
from utils import save_checkpoint, AverageMeter, makedir, get_learning_rate, time_to_str
from losses import eval_f1score
from utils import Logger
from datetime import datetime
from collections import defaultdict
from models.model import BNInception, DPN92, InceptionResNet, DPN68
from lr_scheduler import MultiStepLR
from losses import FocalLoss, BalanceLoss


def train_one_epoch(train_loader, model, criterions, optimizer, epoch, meters, since, log=None):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    if len(meters['f1']):
        previous_loss = meters['loss'][-1]
        previous_f1 = meters['f1'][-1]
        best_f1_epoch = np.argmax(meters['f1'])
        best_f1_score = meters['f1'][best_f1_epoch]
        best_loss_epoch = np.argmin(meters['loss'])
        best_loss = meters['loss'][best_loss_epoch]
    else:
        best_f1_epoch = 0
        best_f1_score = 0
        best_loss_epoch = 0
        best_loss = 0
        previous_loss = 0
        previous_f1 = 0

    for batch_id, (images, target) in enumerate(train_loader):
        batch_x = images.cuda(non_blocking=True)
        target = torch.Tensor(np.array(target)).float().cuda(non_blocking=True)
        output = model(batch_x)
        bce_criterion = criterions[0]
        balance_criterion = criterions[1]
        bce_loss = bce_criterion(output, target)
        balance_loss = balance_criterion(output, target)
        total_loss = bce_loss + 8.0*balance_loss
        losses.update(bce_loss.item(), batch_x.size(0))
        f1_batch = f1_score(target, output.sigmoid().cpu() > 0.15, average='macro')
        f1.update(f1_batch, batch_x.size(0))
        optimizer.zero_grad()
        total_loss.backward()
        # grident clip
        if cfg.grident_clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
        optimizer.step()
        print('Epoch %3d\t' % epoch,
              'Batch %3d|%3d\t' % (batch_id, len(train_loader)),
              'Loss: %10.5f\t' % losses.avg,
              'Metrics|F1 Score: %10.5f\t' % f1.avg,
              'Previous Loss: %10.5f\t' % previous_loss,
              'Previous F1 Score: %10.5f\t' % previous_f1,
              'Best loss:%10.5f Epoch %3d\t' % (best_loss, best_loss_epoch),
              'Besr F1:%10.5f Epoch %3d\t' % (best_f1_score, best_f1_epoch),
              'Time: %s' % time_to_str((timer() - since), 'min'), file=log)

    meters['loss'].append(losses.avg)
    meters['f1'].append(f1.avg)
    return meters


def evaluate(val_loader, model, criterions, epoch, meters, start, log=None):
    losses = AverageMeter()
    f1 = AverageMeter()
    if len(meters['val_f1']):
        best_f1_epoch = np.argmax(meters['val_f1'])
        best_f1_score = meters['val_f1'][best_f1_epoch]
        best_loss_epoch = np.argmin(meters['val_loss'])
        best_loss = meters['val_loss'][best_loss_epoch]
    else:
        best_f1_epoch = 0
        best_f1_score = 0
        best_loss_epoch = 0
        best_loss = 0
    model.cuda()
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_id, (images, target) in enumerate(val_loader):
            batch_x = images.cuda(non_blocking=True)
            batch_y = torch.Tensor(np.array(target)).float().cuda(non_blocking=True)
            output = model(batch_x)
            bce_criterion = criterions[0]
            balance_criterion = criterions[1]
            bce_loss = bce_criterion(output, batch_y)
            balance_loss = balance_criterion(output, batch_y)
            total_loss = bce_loss + balance_loss
            losses.update(bce_loss.item(), batch_x.size(0))
            pred_y = output.sigmoid().cpu().data.numpy()
            preds.append(pred_y)
            targets.append(target)
            f1_batch = f1_score(target, pred_y > 0.15, average='macro')
            f1.update(f1_batch, batch_x.size(0))
            print('Validate Epoch %3d\t' % epoch,
                  'Batch %4d|%4d\t' % (batch_id, len(val_loader)),
                  'Aver Loss: %6.5f\t' % losses.avg,
                  'Aver F1 Score: %6.5f' % f1.avg,
                  'Best Val loss:%10.5f, Epoch: %3d\t' % (best_loss, best_loss_epoch),
                  'Best Val F1:%10.5f, Epoch: %3d\t' % (best_f1_score, best_f1_epoch),
                  'Time: %s' % time_to_str((timer() - start), 'min'), file=log)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    threshold, best_score, std_score = eval_f1score(preds, targets, cfg.label_names, log=log)
    print("Average F1 Score is ", f1.avg, file=log)
    meters['val_loss'].append(losses.avg)
    meters['val_aver_f1'].append(f1.avg)
    meters['val_std_f1'].append(std_score)
    meters['val_f1'].append(best_score)
    meters['threshold'].append(threshold)

    return meters


def submission(test_loader, model, file_name, threshold):
    sub_df = pd.read_csv(cfg.submission_csv)
    filenames, labels, submissions = [], [], []
    model.cuda()
    model.eval()
    for i, (input, filepath) in enumerate(tqdm(test_loader)):
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            labels.append(label > threshold)
            filenames.append(filepath)
    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)

    sub_df['Predicted'] = submissions
    sub_df.to_csv(os.path.join(cfg.submit_dir, file_name), index=None)


def train(task_name, model, optimizer, criterions, scheduler, train_loader, val_loader, log=None):
    meters = defaultdict(list)
    start = timer()
    for epoch in range(0, cfg.epochs):
        scheduler.step(epoch)
        cur_lr = get_learning_rate(optimizer)
        print('Learning rate is ', cur_lr, file=log)
        meters = train_one_epoch(train_loader, model, criterions, optimizer, epoch, meters, start)
        meters = evaluate(val_loader, model, criterions, epoch, meters, start)
        is_best_loss = np.argmin(meters['val_loss']) == epoch
        is_best_f1 = np.argmax(meters['val_f1']) == epoch
        state = {"state_dict": model.state_dict(),
                 "epoch": epoch,
                 "optimizer": optimizer.state_dict(),
                 "meters": meters.copy()}
        save_checkpoint(state, task_name, is_best_loss, is_best_f1)
        print('Task Name: %s\t' % task_name,
              'Validate Epoch %3d\t' % epoch,
              'Train Loss: %6.5f\t' % meters['loss'][-1],
              'Train F1 Score: %6.5f\t' % meters['f1'][-1],
              'Val Loss: %6.5f\t' % meters['val_loss'][-1],
              'Val F1 Score: %6.5f\t' % meters['val_f1'][-1],
              'Val Std F1 Score: %6.5f\t' % meters['val_std_f1'][-1],
              'Val Aver F1 Score: %6.5f\t' % meters['val_aver_f1'][-1],
              'Best Val loss:%6.5f, Epoch: %3d\t' % (np.min(meters['val_loss']), np.argmin(meters['val_loss'])),
              'Best F1 Loss: %6.5f, Epoch: %3d\t' % (np.max(meters['val_f1']), np.argmax(meters['val_f1'])),
              'Time: %s' % time_to_str((timer() - start), 'min'),
              file=log)
        time.sleep(0.01)

    return model


def submission_best_f1(task_name, model, test_loader, log=None):
    best_model = torch.load(os.path.join(cfg.log_dir, task_name, cfg.best_f1_loss_weights))
    meters = best_model["meters"]
    epoch = best_model["epoch"]
    print(meters["threshold"][epoch])
    threshold = 0.50
    model.load_state_dict(best_model["state_dict"])
    print('Best Epoch is ', best_model["epoch"], "\tVal Loss is ", meters["val_loss"][epoch], "\t Val F1 is ",
          meters["val_f1"][epoch], file=log)
    print("Threshold is \n", threshold)
    submission(test_loader, model, task_name + '.csv', threshold)

    return meters


def submission_best_loss(task_name, model, test_loader, log=None):
    best_model = torch.load(os.path.join(cfg.log_dir, task_name, cfg.best_val_loss_weights))
    meters = best_model["meters"]
    epoch = best_model["epoch"]
    threshold = meters["threshold"][epoch]
    threshold = 0.50
    model.load_state_dict(best_model["state_dict"])
    print('Best Epoch is ', best_model["epoch"], "\tVal Loss is ", meters["val_loss"][epoch], "\t Val F1 is ", meters["val_f1"][epoch], file=log)
    print(threshold)
    submission(test_loader, model, task_name + '.csv', threshold)

def task1():
    task_name = "base_dpn62_balance"
    makedir(os.path.join(cfg.log_dir, task_name))
    log = Logger(os.path.join(cfg.log_dir, task_name + '_log.txt'), mode="a")
    log("\n\n" + '-' * 51 + "[START %s]" % datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "-" * 51 + "\n\n")
    print(cfg, file=log)
    train_loader, val_loader, test_loader = get_dataloader()
    model = DPN68()
    model.cuda()
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = BalanceLoss().cuda()
    criterions = [criterion1, criterion2]
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model = train(task_name, model, optimizer, criterions, scheduler, train_loader, val_loader, log=log)
    submission_best_f1(task_name, model, test_loader, log=log)


def task2():
    task_name = "base_dpn62_balance2"
    makedir(os.path.join(cfg.log_dir, task_name))
    log = Logger(os.path.join(cfg.log_dir, task_name + '_log.txt'), mode="a")
    log("\n\n" + '-' * 51 + "[START %s]" % datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "-" * 51 + "\n\n")
    print(cfg, file=log)
    train_loader, val_loader, test_loader = get_dataloader()
    model = DPN92()
    model.cuda()
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = BalanceLoss().cuda()
    criterions = [criterion1, criterion2]
    # optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-5)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 10, 15, 20, 25, 30, 35, 40, 50], gamma=0.4)
    # model = train(task_name, model, optimizer, criterions, scheduler, train_loader, val_loader, log=log)
    submission_best_f1(task_name, model, test_loader, log=log)


if __name__ == "__main__":
    init_env('7')
    makedir(cfg.log_dir)
    makedir(cfg.submit_dir)
    task2()
