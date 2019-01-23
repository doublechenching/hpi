import cv2
from config import config as cfg
from fastai.dataset import FilesDataset
from fastai.transforms import tfms_from_stats, RandomRotate, RandomDihedral, RandomLighting
from fastai.dataset import ImageData, TfmType, A, CropType
import os
import pandas as pd
import numpy as np


def open_image(path, image_id, use_yellow=False):
    colors = ['red', 'green', 'blue', 'yellow']
    img = [cv2.imread(os.path.join(path, image_id+'_'+color+'.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
           for color in colors]

    return np.stack(img, axis=-1)


class HPADataset(FilesDataset):

    def __init__(self,
                 fnames,
                 path,
                 transform,
                 use_yellow=True):
        self.train_df = pd.read_csv(cfg.train_csv).set_index('Id')
        self.train_df['Target'] = [
            [int(i) for i in s.split()] for s in self.train_df['Target']]
        self.use_yellow = use_yellow
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(self.path, self.fnames[i], self.use_yellow)
        if self.sz == 512:
            return img
        else:
            return cv2.resize(img, (self.sz, self.sz), cv2.INTER_AREA)

    def get_y(self, i):
        if(self.path == cfg.test_dir):
            return np.zeros(len(cfg.label_names), dtype=np.int)
        else:
            labels = self.train_df.loc[self.fnames[i]]['Target']
            return np.eye(len(cfg.label_names), dtype=np.float)[labels].sum(axis=0)

    def get_c(self):
        return len(cfg.label_names)

    @property
    def is_multi(self):
        """是否是多标签的样本"""
        return True

    @property
    def is_reg(self):
        """是否是回归模型"""
        return True


def get_data(train_names, val_names, test_names, target_size, batch_size, n_workers=5):
    aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    # std and var
    stats = A([0.08069, 0.05258, 0.05487, 0.08282],
              [0.13704, 0.10145, 0.15313, 0.13814])
    tfms = tfms_from_stats(stats, target_size,
                           crop_type=CropType.NO,
                           tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)

    datasets = ImageData.get_ds(HPADataset,
                                (train_names[:-(len(train_names) % batch_size)], cfg.train_dir),
                                (val_names, cfg.train_dir),
                                tfms,
                                test=(test_names, cfg.test_dir))

    img_ds = ImageData('./',
                       datasets,
                       batch_size,
                       num_workers=n_workers,
                       classes=None)

    return img_ds


def get_dataset_stat(md):
    x_tot = np.zeros(4)
    x2_tot = np.zeros(4)
    for x, y in iter(md.trn_dl):
        tmp =  md.trn_ds.denorm(x).reshape(16,-1)
        x = md.trn_ds.denorm(x).reshape(-1,4)
        x_tot += x.mean(axis=0)
        x2_tot += (x**2).mean(axis=0)
    channel_avr = x_tot/len(md.trn_dl)
    channel_std = np.sqrt(x2_tot/len(md.trn_dl) - channel_avr**2)
    
    return channel_avr, channel_std