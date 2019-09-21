import os
from glob import glob
from os.path import join

import numpy as np
from natsort import natsorted
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.layers import Conv2D
from keras.models import Model

import time
import GPUtil


from Datagen import PngClassDataGenerator, PngDataGenerator


def RenameWeights(file_name):
    # Rename best weights
    h5files = glob('*.h5')
    load_file = max(h5files, key=os.path.getctime)
    os.rename(load_file, file_name)
    print('Renamed weights file {} to {}'.format(
        load_file, file_name))


# datagen parameters
def get_train_params(batch_size, dims, n_channels, shuffle=True):
    return {'batch_size': batch_size,
            'dim': dims,
            'n_channels': n_channels,
            'shuffle': shuffle,
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'brightness_range': None,
            'shear_range': 0.,
            'zoom_range': 0.15,
            'channel_shift_range': 0.,
            'fill_mode': 'constant',
            'cval': 0.,
            'horizontal_flip': True,
            'vertical_flip': False,
            'rescale': None,
            'preprocessing_function': 'CLAHE',
            'interpolation_order': 1}


def get_val_params(batch_size, dims, n_channels, shuffle=False):
    return {'batch_size': batch_size,
            'dim': dims,
            'n_channels': n_channels,
            'shuffle': shuffle,
            'rotation_range': 0,
            'width_shift_range': 0.,
            'height_shift_range': 0.,
            'brightness_range': None,
            'shear_range': 0.,
            'zoom_range': 0.,
            'channel_shift_range': 0.,
            'fill_mode': 'constant',
            'cval': 0.,
            'horizontal_flip': False,
            'vertical_flip': False,
            'rescale': None,
            'preprocessing_function': 'CLAHE',
            'interpolation_order': 1}


def get_class_datagen(pos_datapath, neg_datapath, train_params, val_params, val_split):
    # Get list of files
    positive_img_files = natsorted(glob(join(pos_datapath, '*.png')))
    print('Found {} positive files'.format(len(positive_img_files)))
    negative_img_files = natsorted(
        glob(join(neg_datapath, '*.png')))
    print('Found {} negative files'.format(len(negative_img_files)))

    # make labels
    pos_labels = [1.]*len(positive_img_files)
    neg_labels = [0.]*len(negative_img_files)
    # combine
    pretrain_img_files = positive_img_files + negative_img_files
    pretrain_labels = pos_labels + neg_labels

    # get class weights for  balancing
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(pretrain_labels), pretrain_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Split into test/validation sets
    rng = np.random.RandomState(seed=1)
    pre_trainX, pre_valX, pre_trainY, pre_valY = train_test_split(
        pretrain_img_files, pretrain_labels, test_size=val_split, random_state=rng, shuffle=True)

    pre_train_dict = dict([(f, mf) for f, mf in zip(pre_trainX, pre_trainY)])
    pre_val_dict = dict([(f, mf) for f, mf in zip(pre_valX, pre_valY)])

    # Setup datagens
    train_gen = PngClassDataGenerator(pre_trainX,
                                      pre_train_dict,
                                      **train_params)
    val_gen = PngClassDataGenerator(pre_valX,
                                    pre_val_dict,
                                    **val_params)
    return train_gen, val_gen, class_weight_dict


def get_seg_datagen(img_datapath, mask_datapath, train_params, val_params, val_split):

    train_img_files = natsorted(glob(join(img_datapath, '*.png')))
    train_mask_files = natsorted(glob(join(mask_datapath, '*.png')))

    rng = np.random.RandomState(seed=1)
    # Split into test/validation sets
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_mask_files, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = PngDataGenerator(trainX,
                                 train_dict,
                                 **train_params)
    val_gen = PngDataGenerator(valX,
                               val_dict,
                               **val_params)
    return train_gen, val_gen

def ConvertModelOutputToLinear(model):
    output_layer = model.layers.pop()
    ksize = output_layer.kernel_size
    weights = output_layer.get_weights()
    numF = output_layer.get_weights()[0].shape[-1]
    new_layer = Conv2D(numF,ksize,activation='linear')(model.layers[-1].output)
    newModel = Model(model.input,new_layer)
    newModel.layers[-1].set_weights(weights)
    return newModel


def WaitForGPU(wait=300):
    GPUavailable = False
    while not GPUavailable:
        try:
            if not 'DEVICE_ID' in locals():
                DEVICE_ID = GPUtil.getFirstAvailable()[0]
                print('Using GPU', DEVICE_ID)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
            GPUavailable = True
            return
        except Exception as e:
            # No GPU available
            print('Waiting for GPU...')
            GPUavailable = False
            time.sleep(wait)