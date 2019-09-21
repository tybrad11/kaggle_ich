# %% Setup

'''
input: 
    seg_model_name  (string)  -- determines which model to use:
                                    'resunet', 'block2d'
    model_num  (string)  --  for saving the model 

    skip_positive_only_training (bool)  -- Skip pretraining segmentation model 
                                     on only images that are positive?                                     
    skip_all_cases_training (bool) -- don't train on all cases (the default 
                                    order is all-train, positive-train, 
                                    quality-train)
    skip_quality_only_training (bool)  -- don't train on high quality positive 
                                    cases
    skip_encoder_pretrain (bool) -- skip the pre-training of the classification 
                                   model on NIH data
    skip_encoder_training -- skip any training of classification model/encoder
    




'''

import os
import time
from glob import glob
from os.path import join
import sys

sys.path.append('~/deep_learning/jacobs_git/Kaggle-Pneumothorax')
import GPUtil
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from natsort import natsorted
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from CustomCallbacks import CyclicLR
from Datagen import PngClassDataGenerator, PngDataGenerator
from HelperFunctions import (RenameWeights, WaitForGPU, get_class_datagen,
                             get_seg_datagen, get_train_params, get_val_params)
from Losses import dice_coef_loss
from Models import (BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED, 
                                res_unet, res_unet_encoder, attention_unet, 
                                tiramisu)
from VisTools import DisplayDifferenceMask

import argparse

###########################  Functions  ######################################
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)




def get_seg_model(seg_model_name, input_dims):
    #return segmentation model based on name
    if seg_model_name.lower() == 'block2d':
        full_model = BlockModel2D(input_dims, filt_num=16, numBlocks=4)
    elif seg_model_name.lower() == 'resunet':
        full_model = res_unet(input_dims)
    elif seg_model_name.lower() == 'attunet':
        full_model = attention_unet(input_dims)
    elif seg_model_name.lower() == 'tiramisu':
        full_model = tiramisu(input_dims)
    
    return full_model

def get_classifier_model(seg_model_name, input_dims):
    #return segmentation model based on name
    if seg_model_name.lower() == 'block2d':
        full_model = BlockModel_Classifier(input_shape=input_dims,
                                          filt_num=16, numBlocks=4)
    elif seg_model_name.lower() == 'resunet':
        full_model = res_unet_encoder(input_dims)
    
    return full_model

def get_data_and_train_model(full_model,
                             img_path,
                             mask_path,
                             full_train_params,
                             full_val_params,
                             val_split,
                             best_weight_path,
                             cb_check,
                             learnRate,
                             full_epochs,
                             cb_plateau):        
   
    train_gen, val_gen = get_seg_datagen(
        img_path, mask_path, full_train_params, full_val_params, val_split)
    
    clr_step = 4*len(train_gen)
    cb_clr = CyclicLR(base_lr=learnRate/100, max_lr=learnRate*10,
                      step_size=clr_step, mode='triangular2')
    
    print('-------------------------\n--- Starting training ---\n-------------------------')
    
    # train full size model
    history_full = full_model.fit_generator(generator=train_gen,
                                            epochs=full_epochs, verbose=1,
                                            callbacks=[cb_check, cb_plateau],
                                            validation_data=val_gen)    
    return history_full, full_model, train_gen, val_gen 

##############################################################################
def str2bool(v):   # for parsing booleans -- which are a pain!
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
## parse inputs
parser = argparse.ArgumentParser(description='Model and parameters')
parser.add_argument('seg_model_name', type=str, help='Segmentation model type')
parser.add_argument('model_num', type=str, help='Model save number')
parser.add_argument('positive_only_training', type=str2bool, nargs='?', const=True, default=False, help=' training segmentation model on only images that are positive? (True of False)')
parser.add_argument('all_cases_training', type=str2bool, nargs='?', const=True, default=False, help=' training segmentation model on both positive and negative images? (True of False)')
parser.add_argument('quality_only_training', type=str2bool, nargs='?', const=True, default=False, help=' training segmentation model on only images that are positive and high quality? (True of False)')
parser.add_argument('encoder_pretrain', type=str2bool, nargs='?', const=True, default=False, help=' pretrain ?(True of False)')
parser.add_argument('encoder_training', type=str2bool, nargs='?', const=True, default=False, help='Only train on segmentation images, not classification? (True of False)')
args = parser.parse_args()

seg_model_name = args.seg_model_name
model_num = args.model_num
positive_only_training = args.positive_only_training
all_cases_training = args.all_cases_training
quality_only_training = args.quality_only_training
encoder_pretrain = args.encoder_pretrain
encoder_training = args.encoder_training
  
#print(args.seg_model_name)
#print(args.model_num)
#print(args.positive_only_training)
#print(args.all_cases_training)
#print(args.quality_only_training)
#print(args.encoder_pretrain)
#print(args.encoder_training)


config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)

os.environ['HDF5_USE_FILE_LOCKING'] = 'false'


rng = np.random.RandomState(seed=1)

if False:
    WaitForGPU()


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ SETUP~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~

# Setup data
pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding/'

# use normalized images
# pos_img_path = '/data/Kaggle/pos-norm-png'
# pos_mask_path = '/data/Kaggle/pos-mask-png'

# use clahe-normalized images with only LARGE and POSITIVE masks
pos_img_filt_path = '/home/tjb129/deep_learning/kaggle_pneumothorax/pos-filt-png-clahe'
pos_mask_filt_path = '//home/tjb129/deep_learning/kaggle_pneumothorax/pos-filt-mask-png'

# use clahe-normalized images with (almost) all POSITIVE masks
pos_img_path = '/home/tjb129/deep_learning/kaggle_pneumothorax/pos-all-png-clahe'
pos_mask_path = '/home/tjb129/deep_learning/kaggle_pneumothorax/pos-all-mask-png'

# use clahe-normalized imaged with all masks
all_img_path= '/home/tjb129/deep_learning/kaggle_pneumothorax/train-png-clahe'
all_mask_path= '/home/tjb129/deep_learning/kaggle_pneumothorax/train-mask'

#clahe-normalized postivies rated as high quality
qual_img_path = '/home/tjb129/deep_learning/kaggle_pneumothorax/pos-all-png-quality-clahe'
qual_mask_path =  '/home/tjb129/deep_learning/kaggle_pneumothorax/pos-all-png-quality-mask'

pretrain_weights_filepath = 'Pretrain_class_weights_{}_v{}.h5'
best_weight_filepath = 'Best_Kaggle_Weights_{}_{}_v{}.h5'

# pre-train parameters
pre_im_dims = (512, 512)
pre_n_channels = 1
pre_batch_size = 16
pre_val_split = .15
pre_epochs = 5

# train parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
batch_size_1024 = 1
learnRate = 1e-5
val_split = .15
epochs_unfreeze = [5, 10]  # epochs before and after unfreezing weights
full_epochs = 30  # epochs trained on 1024 data with only large masks
#full_epochs_all = 10  # epochs trained on all positive masks

# model parameters
filt_nums = 16
num_blocks = 4

# datagen params
pre_train_params = get_train_params(
    pre_batch_size, pre_im_dims, pre_n_channels)
pre_val_params = get_val_params(pre_batch_size, pre_im_dims, pre_n_channels)
train_params = get_train_params(batch_size, im_dims, n_channels)
val_params = get_val_params(batch_size, im_dims, n_channels)
full_train_params = get_train_params(batch_size_1024, (1024, 1024), 1)
full_val_params = get_val_params(batch_size_1024, (1024, 1024), 1)

#I've saved the preprocessed data with clahe, so remove it
#pre_train_params["preprocessing_function"] = 'None'
#pre_val_params["preprocessing_function"] = 'None'
train_params["preprocessing_function"] = 'None'
val_params["preprocessing_function"] = 'None'
full_train_params["preprocessing_function"] = 'None'
full_val_params["preprocessing_function"] = 'None'


# %% ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~Pre-training~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~
if encoder_training:
    
    if not encoder_pretrain:
        print('---------------------------------')
        print('---- Setting up pre-training ----')
        print('---------------------------------')
    
        # Get datagens for pre-training
        pre_train_gen, pre_val_gen, class_weights = get_class_datagen(
            pre_train_datapath, pre_train_negative_datapath, pre_train_params, pre_val_params, pre_val_split)
    
        # Create model
        pre_model = get_classifier_model(seg_model_name, pre_im_dims+(pre_n_channels,))

    
        # Compile model
        pre_model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
        # Create callbacks
        cb_check = ModelCheckpoint(pretrain_weights_filepath.format(seg_model_name, model_num), monitor='val_loss',
                                   verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    
        print('---------------------------------')
        print('----- Starting pre-training -----')
        print('---------------------------------')
    
        # Train model
        pre_history = pre_model.fit_generator(generator=pre_train_gen,
                                              epochs=pre_epochs, verbose=1,
                                              callbacks=[cb_check],
                                              class_weight=class_weights,
                                              validation_data=pre_val_gen)
    
        # Load best weights
        pre_model.load_weights(pretrain_weights_filepath.format(seg_model_name, model_num))
    
        # Calculate confusion matrix
        print('Calculating classification confusion matrix...')
        pre_val_gen.shuffle = False
        preds = pre_model.predict_generator(pre_val_gen, verbose=1)
        labels = [pre_val_gen.labels[f] for f in pre_val_gen.list_IDs]
        y_pred = np.rint(preds)
        totalNum = len(y_pred)
        y_true = np.rint(labels)[:totalNum]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
        print('----------------------')
        print('Classification Results')
        print('----------------------')
        print('True positives: {}'.format(tp))
        print('True negatives: {}'.format(tn))
        print('False positives: {}'.format(fp))
        print('False negatives: {}'.format(fn))
        print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
        print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
        print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
        print('-----------------------')
    
    else:
        # Just create model, then load weights
        pre_model = get_classifier_model(seg_model_name, pre_im_dims+(pre_n_channels,))
        # Load best weights
        pre_model.load_weights(pretrain_weights_filepath.format(seg_model_name, model_num))
    
    # %% ~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~ Training ~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~
    
    print('Setting up 512-training')
    
    # convert to segmentation model
    model = ConvertEncoderToCED(pre_model, seg_model_name)
    
    # create segmentation datagens
    # using positive, large mask images only
    train_gen, val_gen = get_seg_datagen(
        pos_img_filt_path, pos_mask_filt_path, train_params, val_params, val_split)
    
    
    # Create callbacks
    best_weight_path = best_weight_filepath.format(seg_model_name,'512train', model_num)
    cb_check = ModelCheckpoint(best_weight_path, monitor='val_loss',
                               verbose=1, save_best_only=True,
                               save_weights_only=True, mode='auto', period=1)
    
    cb_plateau = ReduceLROnPlateau(
        monitor='val_loss', factor=.5, patience=3, verbose=1)
    
    # Compile model
    model.compile(Adam(lr=learnRate), loss=dice_coef_loss)
    
    print('---------------------------------')
    print('----- Starting 512-training -----')
    print('---------------------------------')
    
    history = model.fit_generator(generator=train_gen,
                                  epochs=epochs_unfreeze[0], verbose=1,
                                  callbacks=[cb_plateau],
                                  validation_data=val_gen)
    
    # make all layers trainable again
    for layer in model.layers:
        layer.trainable = True
    
    # Compile model
    model.compile(Adam(lr=learnRate), loss=dice_coef_loss)
    
    print('----------------------------------')
    print('--Training with unfrozen weights--')
    print('----------------------------------')
    
    history2 = model.fit_generator(generator=train_gen,
                                   epochs=epochs_unfreeze[1],
                                   verbose=1,
                                   callbacks=[cb_check, cb_plateau],
                                   validation_data=val_gen)
    
    # %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~ Full Size Training ~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    print('Setting up 1024 training')
    
#    # make full-size model
#    if seg_model_name.lower() == 'blockmodel2d':
    input_dims = (1024, 1024, n_channels)
    full_model = get_seg_model(seg_model_name, input_dims)
    full_model.load_weights(best_weight_path)


else:
    input_dims = (1024, 1024, n_channels)
    full_model = get_seg_model(seg_model_name, input_dims)    

    
    
# Compile model
full_model.compile(Adam(lr=learnRate), loss=dice_coef_loss)

# Set weight paths
best_weight_path = best_weight_filepath.format(seg_model_name,'1024train', model_num)


# Create callbacks
cb_plateau = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=5, verbose=1)
cb_check = ModelCheckpoint(best_weight_path, monitor='val_loss',
                           verbose=1, save_best_only=True,
                           save_weights_only=True, mode='auto', period=1)    
#train

#train with all data
if  all_cases_training:
    print('Training with all data')
    history_full, full_model, train_gen, val_gen  = get_data_and_train_model(full_model, 
                                    all_img_path, all_mask_path, 
                                    full_train_params, full_val_params, val_split, 
                                    best_weight_path, cb_check, learnRate, 
                                    full_epochs, cb_plateau)

#if post-training with only positive masks
if positive_only_training:
    print('Training with only positives...')
    history_full, full_model, train_gen, val_gen  = get_data_and_train_model(full_model, 
                                pos_img_path, pos_mask_path, 
                                full_train_params, full_val_params, val_split, 
                                best_weight_path, cb_check, learnRate, 
                                full_epochs, cb_plateau)


if  quality_only_training:
    print('Training with only quality positives...')
    history_full, full_model, train_gen, val_gen  = get_data_and_train_model(full_model, 
                                qual_img_path, qual_mask_path, 
                                full_train_params, full_val_params, val_split, 
                                best_weight_path, cb_check, learnRate, 
                                full_epochs, cb_plateau)


# %% make some demo images

full_model.load_weights(best_weight_path)

folder_save='sample_difference_images_{}_v{}'.format(seg_model_name, model_num)
if not os.path.exists(folder_save):
    os.mkdir(folder_save)

count = 0
for rep in range(20):
    testX, testY = val_gen.__getitem__(rep)
    preds = full_model.predict_on_batch(testX)

    for im, mask, pred in zip(testX, testY, preds):
        DisplayDifferenceMask(im[..., 0], mask[..., 0], pred[..., 0],
                              savepath=os.path.join(folder_save, 
                                'SampleDifferenceMasks_{}.png'.format(count)))
        count += 1


# %%
