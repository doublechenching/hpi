#encoding: utf-8
from __future__ import print_function
import numpy as np
import os
from config import config as cfg
from training import init_env
cfg.batch_size = 8
from proc.data import load_train_csv, split_train_val, load_test_csv
from model._xception import Xception, preprocess_input
from proc.gennerator import BaseTestGenerator
from sklearn.metrics import roc_curve, auc, f1_score
from matplotlib import pyplot as plt
from config import config as cfg
import scipy.optimize as opt


def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))


def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)

    return score


def fit_val(x, y, classes_num):
    params = 0.5*np.ones(classes_num)
    wd = 1e-5

    def error(p): return np.concatenate(
        (F1_soft(x, y, p) - 1.0, wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)

    return p


def eval_f1score(task_name, pred_Y, test_Y, all_labels):
    log_name = task_name + '_f1score.txt'
    f = open(log_name, 'w')
    threshold = fit_val(pred_Y, test_Y, len(all_labels))
    threshold[threshold < 0.1] = 0.1
    score = f1_score(test_Y, pred_Y > threshold, average='macro')
    pred_pos = np.mean(pred_Y > threshold, axis=0)
    test_pos = np.mean(test_Y, axis=0)
    acc = np.mean((pred_Y > threshold) == test_Y, axis=0)
    print('Total F1-score: ', np.mean(score))
    print('Total F1-score: ', np.mean(score), file=f)
    for idx, name in all_labels.items():
        score_idx = f1_score(test_Y[:, idx], (pred_Y > threshold)[
                             :, idx], average='macro')
        print('%s: Threshold: %.3f, 预测阳性比例: %2.2f%%, 实际阳性比例: %2.2f%%, acc: %2.2f%%, F1-score: %2.2f%%'
              % (name, threshold[idx], pred_pos[idx] * 100, test_pos[idx] * 100, acc[idx] * 100, score_idx * 100))
        print('%s: Threshold: %.3f, 预测阳性比例: %2.2f%%, 实际阳性比例: %2.2f%%, acc: %2.2f%%, F1-score: %2.2f%%'
              % (name, threshold[idx], pred_pos[idx] * 100, test_pos[idx] * 100, acc[idx] * 100, score_idx * 100), file=f)
    f.close()
    return threshold


def show_samples(test_X, test_Y, pred_Y):
    # 表现最差样本样本id
    sickest_idx = np.argsort(np.sum(test_Y, 1) < 1)
    fig, m_axs = plt.subplots(4, 2, figsize=(16, 32))

    for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
        c_ax.imshow(test_X[idx, :, :, 0], cmap='bone')
        stat_str = [n_class[:6] for n_class, n_score in zip(
            cfg.label_names, test_Y[idx]) if n_score > 0.5]
        pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100) for n_class, n_score, p_score in zip(cfg.label_names, test_Y[idx], pred_Y[idx])
                    if (n_score > 0.5) or (p_score > 0.5)]
        c_ax.set_title('Dx: '+', '.join(stat_str) +
                       '\nPDx: '+', '.join(pred_str))
        c_ax.axis('off')

    fig.savefig('trained_img_predictions.png')


def load_val_gennerator():
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)

    val_gen = BaseTestGenerator(val_df, cfg.train_dir,
                                batch_size=cfg.batch_size,
                                aug_args=cfg.aug_args,
                                target_shape=cfg.input_shape[:2],
                                use_yellow=False,
                                return_label=True,
                                preprocessing_function=preprocess_input
                                )

    return val_gen, val_df


def load_train_gennerator():
    train_val_df = load_train_csv(cfg)
    train_df, val_df = split_train_val(train_val_df, 0.25, seed=42)
    train_gen = BaseTestGenerator(train_df, cfg.train_dir,
                                  batch_size=cfg.batch_size,
                                  aug_args=cfg.aug_args,
                                  target_shape=cfg.input_shape[:2],
                                  use_yellow=False,
                                  return_label=True,
                                  preprocessing_function=preprocess_input
                                  )

    return train_gen, train_df


def load_test_gennerator():
    test_df = load_test_csv(cfg)
    test_gen = BaseTestGenerator(test_df, cfg.test_dir,
                                 batch_size=cfg.batch_size,
                                 aug_args=cfg.aug_args,
                                 target_shape=cfg.input_shape[:2],
                                 use_yellow=False,
                                 return_label=False,
                                 preprocessing_function=preprocess_input
                                 )

    return test_gen, test_df


def predict(gen, weights_path, return_label=True):
    model = Xception(input_shape=cfg.input_shape,
                     include_top=True,
                     weights=weights_path,
                     n_class=len(cfg.label_names))
    pred_Y = []
    test_Y = []

    for batch_id in range(len(gen)):
        if return_label:
            batch_x, batch_y = gen[batch_id]
        else:
            batch_x = gen[batch_id]

        batch_pred = model.predict(batch_x, batch_size=len(batch_x))
        batch_pred = np.split(batch_pred, cfg.batch_size, axis=0)

        for i, pred in enumerate(batch_pred):
            # (1, n_classes)
            pred = np.mean(pred, axis=0, keepdims=True)
            pred_Y.append(pred)
            pred_cache = np.round(pred[0, :])
            if return_label:
                score = np.mean(np.round(pred[:]) == batch_y[i, :])
                batch_y_cache = batch_y[i, :]
                print('batch' + str(batch_id + 1) + '_' + str(i), ' prediction: ',
                      np.where(pred_cache > 0)[0], '  gt: ', np.where(batch_y_cache > 0)[0])
                print('predicting batch ', batch_id + 1, ', total',
                      len(gen), '---- accuracy score: ', score)
            else:
                print('batch' + str(batch_id + 1) + '_' + str(i),
                      ' prediction: ', np.where(pred_cache > 0)[0])
                print('predicting batch ', batch_id + 1, ', total', len(gen))

        if return_label:
            test_Y.append(batch_y)

    pred_Y = np.concatenate(pred_Y, axis=0)  # (batch, n_classes)
    if return_label:
        test_Y = np.concatenate(test_Y, axis=0)
        return pred_Y, test_Y
    else:
        return pred_Y


def predict_on_gennerator(gen, weights_path, return_label=True):
    model = Xception(input_shape=cfg.input_shape, include_top=True,
                     weights=weights_path,
                     n_class=len(cfg.label_names))
    pred_Y = []
    batch_pred = model.predict_generator(gen, steps=len(gen),
                                         use_multiprocessing=True,
                                         verbose=1,
                                         workers=5,
                                         max_queue_size=200)
    batch_pred = np.split(batch_pred, gen.test_df.shape[0], axis=0)
    if return_label:
        test_Y = gen.get_all_labels()
    for batch_id, pred in enumerate(batch_pred):
        pred = np.mean(pred, axis=0, keepdims=True)         # (1, n_classes)
        if return_label:
            acc = np.mean(test_Y[batch_id, :] == (pred[0, :] > 0.5).astype(int))
            print('predicting batch ', batch_id + 1, ', total',
                  gen.test_df.shape[0], ' acc: ', acc)
        else:
            print('predicting batch ', batch_id + 1,
                  ', total', gen.test_df.shape[0])
        pred_Y.append(pred)
    pred_Y = np.concatenate(pred_Y, axis=0)  # (batch, n_classes)
    if return_label:
        return pred_Y, test_Y
    else:
        return pred_Y


def eval_roc(pred_Y, test_Y, all_labels):
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for idx, name in all_labels.items():
        fpr, tpr, thresholds = roc_curve(
            test_Y[:, idx].astype(int), pred_Y[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (name, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig(task_name+'_roc.png')

    pred_pos = np.mean(pred_Y, axis=0)
    test_pos = np.mean(test_Y, axis=0)
    for idx, name in all_labels.items():
        acc = np.mean(np.round(pred_Y[:, idx]) == test_Y[:, idx])
        print('%s: 预测阳性比例: %2.2f%%, 实际阳性比例: %2.2f%%, acc: %2.2f' %
              (name, pred_pos[idx]*100, test_pos[idx]*100, acc))


def eval(task_name, epoch, use_multiprocessing=False):
    log_path = os.path.join(cfg.log_dir, task_name)
    weights_path = os.path.join(log_path, cfg.weights_file).format(epoch=epoch)
    print("weights path ----", weights_path)
    val_gen, val_df = load_val_gennerator()
    if use_multiprocessing:
        pred_Y, test_Y = predict_on_gennerator(val_gen, weights_path)
    else:
        pred_Y, test_Y = predict(val_gen, weights_path)
    eval_roc(pred_Y, test_Y, cfg.label_names)
    threshold = eval_f1score(task_name, pred_Y, test_Y, cfg.label_names)

    return threshold


def submission(task_name, epoch, submission_file, threshold, use_multiprocessing=False):
    test_gen, test_df = load_test_gennerator()
    log_path = os.path.join(cfg.log_dir, task_name)
    weights_path = os.path.join(log_path, cfg.weights_file).format(epoch=epoch)
    print("weights path ----", weights_path)
    if use_multiprocessing:
        pred_Y = predict_on_gennerator(test_gen, weights_path, return_label=False)
    else:
        pred_Y = predict(test_gen, weights_path, return_label=False)
    prediction_Y = pred_Y > threshold
    prediction = []
    for pred in prediction_Y:
        str_labels = ''
        for item in np.nonzero(pred)[0]:
            str_labels += str(item) + ' '
        prediction.append(str_labels.strip())
    test_df['Predicted'] = np.array(prediction)
    test_df.to_csv(submission_file, index=False)


if __name__ == "__main__":
    init_env('0')
    epoch = 37
    task_name = 'xception2'
    submission_file = 'submission_' + task_name + '_' + str(epoch) + '.csv'
    threshold = eval(task_name, epoch, use_multiprocessing=True)
    print(threshold)
    submission(task_name, epoch, submission_file, threshold, use_multiprocessing=True)
