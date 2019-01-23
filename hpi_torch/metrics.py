#encoding: utf-8
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import f1_score
from config import config as cfg

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds, targets, th=0.5, d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targets = targets.astype(np.float)
    score = 2.0*(preds*targets).sum(axis=0)/((preds+targets).sum(axis=0) + 1e-6)

    return score

def fit_val(x, y, classes_num):
    params = 0.5*np.ones(classes_num)
    wd = 1e-5
    def error(p): return np.concatenate(
        (F1_soft(x, y, p) - 1.0, wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)

    return p

def Count_soft(preds, th=0.5, d=50.0):
    preds = sigmoid_np(d*(preds - th))
    return preds.mean(axis=0)


def fit_test(x, y):
    params = 0.5*np.ones(len(cfg.label_names))
    wd = 1e-5

    def error(p): return np.concatenate((Count_soft(x, p) - y,
                                         wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    
    return p


def eval_f1score(preds, targets, all_labels, min_threshold=0.1, log=None):
    threshold = fit_val(preds, targets, len(all_labels))
    # 所有极低的threshold都设置为0.1
    print('Origin Threshold is ', threshold)
    threshold[threshold < min_threshold] = min_threshold
    print('Cleaned Threshold is ', threshold)
    pred_pos = np.mean(preds > threshold, axis=0)
    test_pos = np.mean(targets, axis=0)
    acc = np.mean((preds > threshold) == targets, axis=0)
    for idx, name in all_labels.items():
        score_idx = f1_score(targets[:, idx], (preds > threshold)[:, idx], average='macro')
        print('%s: Threshold: %.3f, 预测阳性比例: %2.2f%%, 实际阳性比例: %2.2f%%, acc: %2.2f%%, F1-score: %2.2f%%'
              % (name, threshold[idx], pred_pos[idx] * 100, test_pos[idx] * 100, acc[idx] * 100, score_idx * 100), file=log)
    best_score = f1_score(targets, preds > threshold, average='macro')
    std_score = f1_score(targets, preds > 0.5, average='macro')
    print('Best F1-score Macro: ', best_score, file=log)
    print('F1-score Macro(Threshold=0.5): ', std_score, file=log)

    return threshold, best_score, std_score


def acc(preds, targs, threshold=0.0):
    """计算accuracy

    # Args
        preds: tensor, predict tensor
        targs: tensor, ground turth tensor
        threshold: float, range in [0, 1.0]
    """
    preds = (preds > threshold).int()
    targs = targs.int()
    
    return (preds == targs).float().mean()