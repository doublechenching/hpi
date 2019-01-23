#encoding: utf-8
from PIL import Image
import numpy as np
from config import config as cfg
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa
import pathlib
import cv2
import pandas as pd
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class HumanDataset(Dataset):
    def __init__(self, images_df,
                 base_path,
                 target_shape=(512, 512),
                 augument=True,
                 use_yellow=False,
                 mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=np.arange(0, cfg.num_classes))
        self.mlb.fit(np.arange(0, cfg.num_classes))
        self.mode = mode
        self.target_shape = target_shape
        self.use_yellow = use_yellow

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X = self.read_images(index)
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y = np.eye(cfg.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())

        if self.augument:
            X = self.augumentor(X)

        X = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        
        return X.float(), y

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        if 'ENSG' in filename:
            filename = os.path.split(filename)[-1]
            filename = os.path.join(cfg.extra_data, filename)
            images = np.array(Image.open(filename + ".png"))

        else:
            r = np.array(Image.open(filename + "_red.png"))
            g = np.array(Image.open(filename + "_green.png"))
            b = np.array(Image.open(filename + "_blue.png"))
            images = [r, g, b]
            if self.use_yellow:
                y = np.array(Image.open(filename + "_yellow.png"))
                images.append(y)
            images = np.stack(images, axis=-1)

        images = images.astype(np.uint8)

        if self.target_shape == (512, 512) and images.shape[:2] == (512, 512):
            return images
        else:
            return cv2.resize(images, self.target_shape)

    def augumentor(self, image):
        sometimes = lambda aug: iaa.Sometimes(0.8, aug)
        augment_img = iaa.Sequential([iaa.Fliplr(0.5),
                                      iaa.Flipud(0.5),
                                      iaa.BilateralBlur(),
                                      iaa.Affine(rotate=90),
                                      iaa.ContrastNormalization((0.8, 1.3)),
                                      sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                                           translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                                           rotate=(-30, 30),
                                                           shear=(-5, 5)
                                                           ))
                                      ],
                                     random_order=True)
                                     
        image_aug = augment_img.augment_image(image)

        return image_aug


def get_dataloader(mix_up=False):
    train_df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.submission_csv)
    train_data_list, val_data_list = train_test_split(train_df, test_size=cfg.split_ratio, random_state=42)
    trainset = HumanDataset(train_data_list, cfg.train_data, mode="train")
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.n_workers)
    val_gen = HumanDataset(val_data_list, cfg.train_data, augument=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.n_workers)
    test_gen = HumanDataset(test_df, cfg.test_data, augument=False, mode="test")
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=cfg.n_workers)

    if mix_up:
        mix_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.n_workers)
        return train_loader, val_loader, test_loader, mix_loader
    else:
        return train_loader, val_loader, test_loader


def get_kfold_dataloader(k=5, n_select=0, use_extra=True, target_shape=(512, 512)):
    kf = KFold(k, random_state=42)
    train_df = pd.read_csv(cfg.train_csv)
    if use_extra:
        extra_df = pd.read_csv(cfg.extra_csv)
        print('number of extra data is ', len(extra_df), '\tnumber of hpi data is ', len(train_df))
        train_df = pd.concat([train_df, extra_df], axis=0)
        print('after concat, number of train data is ', len(train_df))

    test_df = pd.read_csv(cfg.submission_csv)
    print('trainset split ', kf.get_n_splits(train_df), 'folds')
    train_val_groups = []
    for train_index, val_index in kf.split(train_df):
        print('length of trainset is ', len(train_index), '\tlength of valset is ', len(val_index))
        train_val_groups.append([train_df.iloc[train_index], train_df.iloc[val_index]])

    train_data_list, val_data_list = train_val_groups[n_select]
    trainset = HumanDataset(train_data_list, cfg.train_data, mode="train", target_shape=target_shape)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.n_workers)

    val_gen = HumanDataset(val_data_list, cfg.train_data, augument=False, mode="train", target_shape=target_shape)
    val_loader = DataLoader(val_gen, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.n_workers)

    test_gen = HumanDataset(test_df, cfg.test_data, augument=False, mode="test", target_shape=target_shape)
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=cfg.n_workers)

    return train_loader, val_loader, test_loader
