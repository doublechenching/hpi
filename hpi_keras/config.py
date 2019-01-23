#encoding: utf-8
from __future__ import print_function
import os
import platform
from utils import makedir

class Struct:
    """pesudo struct
    """

    def __init__(self):
        pass

    def __str__(self):
        print("net work config")
        print("*"*80)
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        return '\n'.join(str_list)


config = Struct()
if 'Win' in platform.system():
    print('using data from windows')
    config.data_dir = 'V:/data_repos/hpi'
    
else:
    print('using data from linux')
    config.data_dir = '/home/share/data_repos/hpi'

config.extra_data_dir = os.path.join(config.data_dir, 'extra')
config.extra_data_csv = os.path.join(config.extra_data_dir, 'subcellular_location.tsv')

config.train_dir = os.path.join(config.data_dir, 'train')
config.test_dir = os.path.join(config.data_dir, 'test')
config.train_csv = os.path.join(config.data_dir, 'train.csv')
config.test_sample_csv = os.path.join(config.data_dir, 'sample_submission.csv')
config.test_csv = os.path.join(config.data_dir, 'submission.csv')
config.log_dir = './logs'
config.input_shape = [512, 512, 3]
config.epochs = 100
config.n_works = 5
config.n_queue = 100
config.val_steps = 'auto'
config.batch_size = 8
config.weights_file = 'train_epoch_{epoch:02d}.hdf5'
config.random_seed = 42
config.patience = 5
config.inception_imagenet_weights = os.path.join(config.log_dir, 'inception_resnet.h5')

config.label_names = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}

config.aug_args = {'featurewise_normalization': False,
                   'samplewise_normalization': False,
                   'horizontal_flip': True,
                   'vertical_flip': True,
                   'height_shift_range': 0.1,
                   'width_shift_range': 0.1,
                   'rotation_range': 20,
                   'shear_range': 0.1,
                   'fill_mode': 'constant',
                   'cval': 0,
                   'zoom_range': [0.7, 1.5],
                   'equalize': True,
                   'rot90': True
                   }


makedir(config.log_dir)
if __name__ == "__main__":
    print(config.aug_args['equalize'])
    for idx, name in config.label_names.items():
        print(idx, name)
    print(config.weights_file.format(epoch=5))
