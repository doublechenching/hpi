#encoding: utf-8
from __future__ import print_function
import os
import platform

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
    config.data_dir = 'K:/hpi'

else:
    print('using data from linux')
    config.data_dir = '/home/share/data_repos/hpi'

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

config.log_dir = './logs'
config.train_data = os.path.join(config.data_dir, 'train')
config.test_data = os.path.join(config.data_dir, 'test')
config.extra_data = os.path.join(config.data_dir, 'extra', 'external_data')
config.extra_csv = os.path.join(config.data_dir, 'extra', 'extra_data.csv')
config.train_csv = os.path.join(config.data_dir, 'train.csv')
config.submission_csv = os.path.join(config.data_dir, 'sample_submission.csv')
config.weights_file = 'train_epoch_{epoch:02d}.pt.tar'
config.best_val_loss_weights = 'best_loss_model.pt.tar'
config.best_f1_loss_weights = 'best_f1_model.pt.tar'
config.submit_dir = "./submit"
config.num_classes = 28
config.img_width = 512
config.img_height = 512
config.channels = 3
config.lr = 1e-3
config.batch_size = 8
config.epochs = 100
config.n_workers = 5
config.grident_clip = True
config.split_ratio = 0.2