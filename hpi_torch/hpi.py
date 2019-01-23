#encoding: utf-8
from config import config as cfg
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from pretrainedmodels import dpn92
from data.data_fastai import get_data
from torch import optim
from losses import FocalLossWithLogits
from metrics import acc
from training import ConvLearner
from utils import sigmoid_np, fit_val, fit_test
from sklearn.metrics import f1_score
import warnings

def display_imgs(x):
    """
    Example:
        display_imgs(np.asarray(img_ds.trn_ds.denorm(batch_x)))
    """
    cols = 4
    batch_size = x.shape[0]
    rows = min((batch_size+3)//4, 4)
    fig = plt.figure(figsize=(cols*4, rows*4))
    for i in range(rows):
        for j in range(cols):
            idx = i + j * cols
            fig.add_subplot(rows, cols, idx+1)
            plt.axis('off')
            plt.imshow((x[idx, :, :, :3]*255).astype(np.int))

    plt.show()


def find_lr(learner):
    learner.lr_find()
    learner.sched.plot()
    plt.show()


def train(learner, lr=5e-4, save_name='base_dpn92'):
    learner.fit(lr, 1)
    learner.unfreeze()
    lrs = np.array([lr/10, lr/3, lr])
    learner.fit(lrs/4, 4, cycle_len=2, use_clr=(10, 20))
    learner.fit(lrs/4, 2, cycle_len=4, use_clr=(10, 20))
    learner.fit(lrs/16, 1, cycle_len=8, use_clr=(5, 20))
    learner.save(save_name)

    return learner


def save_pred(pred, test_names, th=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)

    sample_df = pd.read_csv(cfg.test_sample_csv)
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value) in zip(test_names, pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id': sample_list, 'Predicted': pred_list_cor})

    df.to_csv(fname, header=True, index=False)


def get_val_threshold(learner):
    preds, y = learner.TTA(n_aug=4)
    preds = np.stack(preds, axis=-1)
    print(preds.shape)
    preds = sigmoid_np(preds)
    pred = preds.max(axis=-1)
    th = fit_val(pred, y)
    th[th < 0.1] = 0.1
    print('Thresholds: ', th)
    print('F1 macro: ', f1_score(y, pred > th, average='macro'))
    print('F1 macro (th = 0.5): ', f1_score(y, pred > 0.5, average='macro'))
    print('F1 micro: ', f1_score(y, pred > th, average='micro'))
    print('Fractions: ', (pred > th).mean(axis=0))
    print('Fractions (true): ', (y > th).mean(axis=0))

    return th


def get_test_threshold(pred_t, prob, min_th=0.1):
    th_t = fit_test(pred_t, prob)
    th_t[th_t < min_th] = min_th
    print('Thresholds: ', th_t)
    print('Fractions: ', (pred_t > th_t).mean(axis=0))
    print('Fractions (th = 0.5): ', (pred_t > 0.5).mean(axis=0))

    return th_t


def get_dataset_fraction(df):
    label_count = np.zeros(len(cfg.label_names))
    for label in df['Target']:
        l = [int(i) for i in label.split()]
        label_count += np.eye(len(cfg.label_names))[l].sum(axis=0)
    label_fraction = label_count.astype(np.float)/len(df)

    return label_count, label_fraction


def get_brute_threshold(preds):
    thresholds = np.linspace(0, 1, 1000)
    n_classes = len(cfg.label_names)
    score = 0.0
    test_threshold = 0.5*np.ones(n_classes)
    best_threshold = np.zeros(n_classes)
    best_val = np.zeros(n_classes)
    for i in range(n_classes):
        for threshold in thresholds:
            test_threshold[i] = threshold
            score = f1_score(preds > 0.5, preds > test_threshold, average='macro')
            if score > best_val[i]:
                best_threshold[i] = threshold
                best_val[i] = score

        print("Threshold[%d] %0.6f, F1: %0.6f" % (i, best_threshold[i], best_val[i]))
        
        test_threshold[i] = best_threshold[i]

    print("Best threshold: \n", best_threshold)
    print("Best f1:\n", best_val)

    return best_threshold


def main():
    warnings.filterwarnings('ignore')
    train_val_names = list({f[:36] for f in os.listdir(cfg.train_dir)})
    test_names = list({f[:36] for f in os.listdir(cfg.test_dir)})
    train_names, val_names = train_test_split(train_val_names, test_size=0.1, random_state=42)
    batch_size = cfg.batch_size
    target_size = 512
    img_ds = get_data(train_names, val_names, test_names, target_size, batch_size, n_workers=5)
    learner = ConvLearner.pretrained(dpn92, img_ds, ps=[0.5])  # use dropout 50%
    learner.opt_fn = optim.Adam
    learner.clip = 1.0
    learner.crit = FocalLoss()
    learner.metrics = [acc]
    # print(learner.summary())
    train(learner, lr=5e-4, save_name='base_dpn')
    val_th = get_val_threshold(learner)
    # TTA
    preds_t, y_t = learner.TTA(n_aug=4, is_test=True)
    preds_t = np.stack(preds_t, axis=-1)
    preds_t = sigmoid_np(preds_t)
    pred_t = preds_t.max(axis=-1)  # max works better for F1 macro score
    test_names = learner.data.test_ds.fnames
    save_pred(pred_t, test_names, val_th, 'protein_classification_v.csv')

    man_th = np.array([0.565, 0.39, 0.55, 0.345, 0.33, 0.39, 0.33, 0.45, 0.38, 0.39,
                       0.34, 0.42, 0.31, 0.38, 0.49, 0.50, 0.38, 0.43, 0.46, 0.40,
                       0.39, 0.505, 0.37, 0.47, 0.41, 0.545, 0.32, 0.1])
    print('Fractions: ', (pred_t > man_th).mean(axis=0))
    save_pred(pred_t, test_names, man_th, 'protein_classification.csv')

    lb_prob = [0.362397820, 0.043841336, 0.075268817, 0.059322034, 0.075268817,
               0.075268817, 0.043841336, 0.075268817, 0.010000000, 0.010000000,
               0.010000000, 0.043841336, 0.043841336, 0.014198783, 0.043841336,
               0.010000000, 0.028806584, 0.014198783, 0.028806584, 0.059322034,
               0.010000000, 0.126126126, 0.028806584, 0.075268817, 0.010000000,
               0.222493880, 0.028806584, 0.010000000]
    test_th = get_test_threshold(pred_t, lb_prob, min_th=0.1)
    save_pred(pred_t, test_names, test_th, 'protein_classification_f.csv')
    
    save_pred(pred_t, test_names, 0.5, 'protein_classification_05.csv')
    
    label_count, label_fraction = get_dataset_fraction(pd.read_csv(cfg.train_csv).set_index('Id'))
    train_th = get_test_threshold(pred_t, label_fraction, min_th=0.05)
    save_pred(pred_t, test_names, train_th, 'protein_classification_t.csv')

    brute_th = get_brute_threshold(pred_t)
    save_pred(pred_t, test_names, brute_th, 'protein_classification_b.csv')


if __name__ == "__main__":
    main()