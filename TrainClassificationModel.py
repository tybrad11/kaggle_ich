# %% Setup
import os
import time
from glob import glob
from os.path import join

import GPUtil
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from natsort import natsorted
from prettytable import PrettyTable
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from Datagen import PngClassDataGenerator, PngDataGenerator
from HelperFunctions import (RenameWeights, get_class_datagen, get_seg_datagen,
                             get_train_params, get_val_params)
from Losses import dice_coef_loss
from Models import Inception_model, densenet_model, efficient_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)


rng = np.random.RandomState(seed=1)

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ SETUP~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~

class_model_name = 'efficientnet'  # 'inception' or 'efficientnet or 'densenet'
model_version = '2'

# Setup data
# pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax_norm/'
# pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding_norm/'
pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding/'

# train_pos_datapath = '/data/Kaggle/pos-norm-png'
# train_neg_datapath = '/data/Kaggle/neg-norm-png'
train_pos_datapath = '/data/Kaggle/pos-filt-png'
train_neg_datapath = '/data/Kaggle/neg-filt-png'

pretrain_weights_filepath = 'Best_pretrain_class_weights_{}_{}.h5'
train_weights_filepath = 'Best_Kaggle_Classification_Weights_{}_{}_v{}.h5'

# pre-train parameters
pre_im_dims = (512, 512)
pre_n_channels = 1
pre_batch_size = 8
pre_val_split = .15
pre_epochs = 30
pre_multi_process = False
skip_pretrain = False

# train parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
learnRate = 1e-4
filt_nums = 16
num_blocks = 5
val_split = .15
epochs = 10
full_epochs = 55  # epochs trained on 1024 data
multi_process = False

# datagen params
pre_train_params = get_train_params(
    pre_batch_size, pre_im_dims, pre_n_channels)
pre_val_params = get_val_params(pre_batch_size, pre_im_dims, pre_n_channels)
train_params = get_train_params(batch_size, im_dims, n_channels)
val_params = get_val_params(batch_size, im_dims, n_channels)
full_train_params = get_train_params(2, (1024, 1024), 1)
full_val_params = get_val_params(2, (1024, 1024), 1)

# %% ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~Pre-training~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

if not skip_pretrain:

    print('---------------------------------')
    print('---- Setting up pre-training ----')
    print('---------------------------------')

    # Get datagens for pre-training
    pre_train_gen, pre_val_gen, class_weights = get_class_datagen(
        pre_train_datapath, pre_train_negative_datapath, pre_train_params, pre_val_params, pre_val_split)

    # Create model
    if class_model_name.lower() == 'densenet':
        model = densenet_model(input_shape=pre_im_dims+(pre_n_channels,))
    elif class_model_name.lower() == 'inception':    
        model = Inception_model(input_shape=pre_im_dims+(pre_n_channels,))
    elif class_model_name.lower() == 'efficientnet':
        model = efficient_model(input_shape=pre_im_dims+(pre_n_channels,))

    # Compile model
    model.compile(Adam(lr=learnRate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Create callbacks
    cb_plateau = ReduceLROnPlateau(
        monitor='val_loss', factor=.5, patience=10, verbose=1)
    cb_check = ModelCheckpoint(pretrain_weights_filepath.format(class_model_name, model_version), monitor='val_loss',
                               verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    print('---------------------------------')
    print('----- Starting pre-training -----')
    print('---------------------------------')

    # Train model
    pre_history = model.fit_generator(generator=pre_train_gen,
                                      epochs=pre_epochs, use_multiprocessing=pre_multi_process,
                                      workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                      class_weight=class_weights,
                                      validation_data=pre_val_gen)

    # Load best weights
    model.load_weights(pretrain_weights_filepath.format(class_model_name, model_version))

    # Calculate confusion matrix
    print('Calculating classification confusion matrix...')
    pre_val_gen.shuffle = False
    preds = model.predict_generator(pre_val_gen, verbose=1)
    labels = [pre_val_gen.labels[f] for f in pre_val_gen.list_IDs]
    y_pred = np.rint(preds)
    totalNum = len(y_pred)
    y_true = np.rint(labels)[:totalNum]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('--------------------------------------')
    print('Classification Results on pre-training')
    print('--------------------------------------')
    print('True positives: {}'.format(tp))
    print('True negatives: {}'.format(tn))
    print('False positives: {}'.format(fp))
    print('False negatives: {}'.format(fn))
    print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
    print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
    print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
    print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
    print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fp)))
    print('-----------------------')

else:
    # skip pretraining, load weights and go to regular training
    print('Skipping pre-training, setting up model')
    # Create model
     # Create model
    if class_model_name.lower() == 'densenet':
        model = densenet_model(input_shape=pre_im_dims+(pre_n_channels,))
    elif class_model_name.lower() == 'inception':    
        model = Inception_model(input_shape=pre_im_dims+(pre_n_channels,))
    elif class_model_name.lower() == 'efficientnet':
        model = efficient_model(input_shape=pre_im_dims+(pre_n_channels,))
    # Compile model
    model.compile(Adam(lr=learnRate), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(pretrain_weights_filepath.format(class_model_name, model_version))

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ 512 Training~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('---------------------------------')
print('---- Setting up 512 training ----')
print('---------------------------------')

# Get datagens for training
train_gen, val_gen, class_weights = get_class_datagen(
    train_pos_datapath, train_neg_datapath, train_params, val_params, val_split)

# Create callbacks
cur_weights_path = train_weights_filepath.format(class_model_name, '512train', model_version)
cb_plateau = ReduceLROnPlateau(
        monitor='val_loss', factor=.5, patience=5, verbose=1)
cb_check = ModelCheckpoint(cur_weights_path, monitor='val_loss', verbose=1,
                           save_best_only=True, save_weights_only=True, mode='auto', period=1)

print('---------------------------------')
print('----- Starting 512 training -----')
print('---------------------------------')

# Train model
history = model.fit_generator(generator=train_gen,
                              epochs=epochs, use_multiprocessing=multi_process,
                              workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                              class_weight=class_weights,
                              validation_data=val_gen)

# Load best weights
model.load_weights(cur_weights_path)

# Calculate confusion matrix
print('Calculating classification confusion matrix...')
val_gen.shuffle = False
preds = model.predict_generator(val_gen, verbose=1)
labels = [val_gen.labels[f] for f in val_gen.list_IDs]
y_pred = np.rint(preds)
totalNum = len(y_pred)
y_true = np.rint(labels)[:totalNum]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print('--------------------------------------')
print('Classification Results on 512 training')
print('--------------------------------------')
print('True positives: {}'.format(tp))
print('True negatives: {}'.format(tn))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fp)))
print('-----------------------')

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ 1024 Training~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('----------------------------------')
print('---- Setting up 1024 training ----')
print('----------------------------------')

# rebuild model
if class_model_name.lower() == 'densenet':
        full_model = densenet_model(input_shape=(1024, 1024)+(n_channels,))
elif class_model_name.lower() == 'inception':    
        full_model = Inception_model(input_shape=(1024, 1024)+(n_channels,))
elif class_model_name.lower() == 'efficientnet':
        full_model = efficient_model(input_shape=(1024, 1024)+(n_channels,))


full_model.load_weights(cur_weights_path)

# Compile model
full_model.compile(Adam(lr=learnRate), loss='binary_crossentropy',
                   metrics=['accuracy'])

# Get datagens for training
full_train_gen, full_val_gen, class_weights = get_class_datagen(
    train_pos_datapath, train_neg_datapath, full_train_params, full_val_params, val_split)

# Create callbacks
cur_weights_path = train_weights_filepath.format(class_model_name, '1024train', model_version)
cb_check = ModelCheckpoint(cur_weights_path, monitor='val_loss', verbose=1,
                           save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
        monitor='val_loss', factor=.5, patience=5, verbose=1)

print('----------------------------------')
print('----- Starting 1024 training -----')
print('----------------------------------')

# Train model
history = full_model.fit_generator(generator=full_train_gen,
                                   epochs=full_epochs, use_multiprocessing=multi_process,
                                   workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                   class_weight=class_weights,
                                   validation_data=full_val_gen)

# Load best weights
full_model.load_weights(cur_weights_path)

# Calculate confusion matrix
print('Calculating classification confusion matrix...')
full_val_gen.shuffle = False
preds = full_model.predict_generator(full_val_gen, verbose=1)
labels = [full_val_gen.labels[f] for f in full_val_gen.list_IDs]
y_pred = np.rint(preds)
totalNum = len(y_pred)
y_true = np.rint(labels)[:totalNum]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print('---------------------------------------')
print('Classification Results on 1024 training')
print('---------------------------------------')
print('True positives: {}'.format(tp))
print('True negatives: {}'.format(tn))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fp)))
print('-----------------------')


# Make ROC curve
fpr, tpr, thresholds = roc_curve(y_true, preds, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for pneumothorax')
plt.legend(loc="lower right")
plt.show()

# print threshold table
table = PrettyTable(['Threshold', 'True Positive Rate', 'False Positive Rate'])
for t, tp, fp in zip(thresholds, tpr, fpr):
    table.add(['{:.034f}'.format(t), '{:.034f}'.format(
        tp), '{:.034f}'.format(fp)])
print(table)


# Get and display a few predictions
# for ind in range(5):
#     b_ind = np.random.randint(0, len(full_val_gen))
#     valX, valY = full_val_gen.__getitem__(b_ind)
#     preds = full_model.predict_on_batch(valX)
#     cur_im = valX[0]
#     disp_im = np.concatenate([cur_im[..., c]
#                               for c in range(cur_im.shape[-1])], axis=1)
#     plt.imshow(disp_im, cmap='gray')
#     plt.title('Predicted: {} Actual: {}'.format(preds[0], valY[0]))
#     plt.show()
