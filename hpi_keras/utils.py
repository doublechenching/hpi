#encoding: utf-8
from __future__ import print_function
from matplotlib import pyplot as plt
import os
import glob

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print('exist folder---', path)


def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
            
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]


def history_plot(history, save_path=''):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def find_best_weights(path):
    weights_list = glob.glob(os.path.join(path, 'train_epoch*.hdf5'))
    weights_list = sorted(weights_list, key=lambda x: int(os.path.split(x)[-1][12:-5]))
    print('best weights path is ', weights_list[-1])
    
    return weights_list[-1]