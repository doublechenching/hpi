#encoding: utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split


def describe_data(df):
    """analyse class number of dataset

    # Args:

    """
    target_counts = df.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
    print('%-30s' % 'Class', '\t', 'Count', '\t%s' % 'Percentage')

    for value , percent, item in zip(target_counts.ravel(), target_counts.ravel() / np.sum(target_counts.ravel()), target_counts.index.values):
        print('%-30s' % item, '\t', value, '\t%.2f' % percent)


def load_train_csv(cfg):
    """split dataframe using offical train_val and test spliting text

    # Args:
        csv_path: dataset path
    """
    def fill_targets(row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = cfg.label_names[int(num)]
            row.loc[name] = 1
        return row

    train_labels = pd.read_csv(cfg.train_csv)
    for key in cfg.label_names.keys():
        train_labels[cfg.label_names[key]] = 0

    train_labels = train_labels.apply(fill_targets, axis=1)

    print('*'*100)
    describe_data(train_labels)

    return train_labels


def load_extra_csv(cfg):
    """split dataframe using offical train_val and test spliting text

    # Args:
        csv_path: dataset path
    """
    def fill_targets(row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = cfg.label_names[int(num)]
            row.loc[name] = 1
        return row

    train_labels = pd.read_csv(cfg.extra_data_csv)
    for key in cfg.label_names.keys():
        train_labels[cfg.label_names[key]] = 0

    train_labels = train_labels.apply(fill_targets, axis=1)

    print('*'*100)
    describe_data(train_labels)

    return train_labels

def load_test_csv(cfg):
    test_df = pd.read_csv(cfg.test_sample_csv)

    return test_df


def split_train_val(train_val_df, ratio=0.25, seed=42):
    """random split train_val dataframe, note that this method is not patient-wise spliting
    
    # Args:
        train_val_df: dataframe, training and validation dataframe
        ratio: float, spliting ratio
        seed: int, set a random seed, get same reslut every time
    """
    train_df, valid_df = train_test_split(train_val_df,
                                          test_size=ratio,
                                          random_state=seed)
    print('*'*100)
    describe_data(train_df)
    print('*'*100)
    describe_data(valid_df)
    return train_df, valid_df


if __name__ == "__main__":
    pass