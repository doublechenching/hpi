#encoding: utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import range
import warnings
from keras import utils as keras_utils
from skimage.transform import resize
from keras import backend as K
from skimage.io import imread
from skimage.exposure import equalize_adapthist
import os
import random
import math
from .image import ImageTransformer
import numpy as np
import glob

class BaseGenerator(keras_utils.Sequence):
    """Base class for image data iterators.

    # Args
        n: int, num of samples.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, if shuttfe sample index in epoch end.
            if set it True, instance is ordered iterator
        seed: Random seeding for data shuffling.

    """
    def __init__(self, train_df,
                 train_img_dir,
                 batch_size,
                 aug_args=None,
                 seed=42,
                 shuffle=False,
                 target_shape=(512, 512),
                 use_yellow=False,
                 preprocessing_function=None):

        self.train_img_dir = train_img_dir
        self.aug_args = aug_args.copy()
        self.equalize = self.aug_args['equalize']
        self.rot90 = self.aug_args['rot90']
        aug_args.pop('equalize')
        aug_args.pop('rot90')
        self.transformer = ImageTransformer(**aug_args)
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

        self.train_df = train_df
        self.preprocessing_function = preprocessing_function
        self.indexes = np.arange(self.train_df.shape[0])
        self.use_yellow = use_yellow


    def __getitem__(self, index):
        batch_size = self.batch_size
        batch_indexes = self.indexes[index*batch_size:(index+1)*batch_size]

        batch_x, batch_y = self.data_generation(batch_indexes)
        
        return batch_x, batch_y


    def __len__(self):

        return math.ceil(self.train_df.shape[0] / float(self.batch_size))


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def data_generation(self, batch_indexes):
        batch_image = []
        batch_label = []
        
        for idx in batch_indexes:

            imgs, label = self.data_load(idx)
            image = self.data_aug(imgs)

            if self.preprocessing_function:
                image = self.preprocessing_function(image, data_format='channels_last')

            batch_image.append(image)
            batch_label.append(label)

        return np.array(batch_image), np.array(batch_label)


    def data_load(self, idx):
        img_id = self.train_df.iloc[idx, 0]
        label = np.array(self.train_df.iloc[idx, 2:], dtype='float32')
        img_r = imread(os.path.join(self.train_img_dir, img_id+'_red.png'))
        img_g = imread(os.path.join(self.train_img_dir, img_id+'_green.png'))
        img_b = imread(os.path.join(self.train_img_dir, img_id+'_blue.png'))
        imgs = [img_r, img_g, img_b]
        if self.use_yellow:
            img_y = imread(os.path.join(self.train_img_dir, img_id+'_yellow.png'))
            imgs.append(img_y)
        
        return imgs, label


    def data_aug(self, imgs):
        if not self.target_shape == (512, 512):
            re_imgs = []
            for img in imgs:
                img = resize(img, self.target_shape[:2], preserve_range=True)
                re_imgs.append(img)

            imgs = re_imgs

        if self.equalize:
            eq_imgs = []
            for img in imgs:
                img = equalize_adapthist(img / 255) * 255
                eq_imgs.append(img)

            imgs = eq_imgs

        image = np.stack(imgs, axis=-1)
        if self.rot90:
            image = np.rot90(image, axes=(0, 1))

        image_shape = list(self.target_shape[:2]) + [3]

        if self.use_yellow:
            image_shape[-1] = 4

        params = self.transformer.get_random_transform(image_shape)
        image = self.transformer.apply_transform(image.astype(K.floatx()), params, order=1)

        return image


class BaseTestGenerator(keras_utils.Sequence):
    """Base class for image data iterators.

    # Args
    """
    def __init__(self, test_df, test_img_dir, batch_size, 
                 aug_args=None, target_shape=(512, 512), 
                 use_yellow=False,
                 return_label=False,
                 preprocessing_function=None):

        self.test_img_dir = test_img_dir
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.aug_args = aug_args.copy()
        self.test_df = test_df
        self.preprocessing_function = preprocessing_function
        self.indexes = np.arange(self.test_df.shape[0])
        self.use_yellow = use_yellow
        self.return_label = return_label


    def __getitem__(self, index):
        batch_size = self.batch_size
        self.batch_indexes = self.indexes[index*batch_size:(index+1)*batch_size]
        self.batch_x, self.batch_y = self.data_generation(self.batch_indexes)
        
        if self.return_label:
            return self.batch_x, self.batch_y
        else:
            return self.batch_x

    def __len__(self):

        return math.ceil(self.test_df.shape[0] / self.batch_size)


    def data_generation(self, batch_indexes):
        batch_image = []
        batch_label = []
        
        for idx in batch_indexes:
            imgs, label = self.data_load(idx)
            image = self.tta(imgs)
            if self.preprocessing_function:
                image = self.preprocessing_function(image, data_format='channels_last')
            batch_image.append(image)
            batch_label.append(label)
            
        return np.concatenate(batch_image, axis=0), np.array(batch_label)


    def data_load(self, idx):
        img_id = self.test_df.iloc[idx, 0]
        label = None

        if self.return_label:
            label = np.array(self.test_df.iloc[idx, 2:], dtype='float32')
        
        img_r = imread(os.path.join(self.test_img_dir, img_id+'_red.png'))
        img_g = imread(os.path.join(self.test_img_dir, img_id+'_green.png'))
        img_b = imread(os.path.join(self.test_img_dir, img_id+'_blue.png'))
        imgs = [img_r, img_g, img_b]

        if self.use_yellow:
            img_y = imread(os.path.join(self.test_img_dir, img_id+'_yellow.png'))
            imgs.append(img_y)
        
        return imgs, label


    def tta(self, imgs):

        if not self.target_shape == (512, 512):
            re_imgs = []
            for img in imgs:
                img = resize(img, self.target_shape[:2], preserve_range=True)
                re_imgs.append(img)
            imgs = re_imgs

        if self.aug_args['equalize']:
            eq_imgs = []
            for img in imgs:
                img = equalize_adapthist(img / 255) * 255
                eq_imgs.append(img)
            imgs = eq_imgs

        image = np.stack(imgs, axis=-1)
        test_images = []
        
        if self.aug_args['horizontal_flip']:
            test_images.append(image)
            test_images.append(np.flip(image, axis=1))

        if self.aug_args['vertical_flip']:
            aug_imgs = []
            for image in test_images:
                aug_imgs.append(np.flip(image, axis=0))
            test_images += aug_imgs

        if self.aug_args['rot90']:
            aug_imgs = []
            for image in test_images:
                aug_imgs.append(np.rot90(image, axes=(0, 1)))
            test_images += aug_imgs

        test_images = np.stack(test_images, axis=0)

        return test_images

    def get_all_labels(self):
        labels = []
        for idx in self.indexes:
            label = np.array(self.test_df.iloc[idx, 2:], dtype='float32')
            labels.append(label)

        return np.array(labels)


class BaseExtraDataGenerator(BaseGenerator):

    def data_load(self, idx):
        img_id = self.train_df.iloc[idx, 0]
        label = np.array(self.train_df.iloc[idx, 2:], dtype='float32')
        imgs = imread(glob.glob(os.path.join(self.train_img_dir, img_id+'*.png'))[0])
        imgs = [imgs[0], imgs[1], imgs[2]]
        
        return imgs, label

