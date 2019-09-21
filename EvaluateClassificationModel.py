# %% Setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
from glob import glob
from os.path import join

import GPUtil
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from Datagen import PngClassDataGenerator, PngDataGenerator
from HelperFunctions import (RenameWeights, get_class_datagen, get_seg_datagen,
                             get_train_params, get_val_params)
from Losses import dice_coef_loss
from Models import Inception_model


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

# Setup data
pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding/'

# train_pos_datapath = '/data/Kaggle/pos-norm-png'
# train_neg_datapath = '/data/Kaggle/neg-norm-png'
train_pos_datapath = '/data/Kaggle/pos-filt-png'
train_neg_datapath = '/data/Kaggle/neg-filt-png'

# parameters
im_dims = (1024, 1024)
n_channels = 1
batch_size = 4
learnRate = 1e-4
filt_nums = 16
num_blocks = 5
val_split = .15
train_weights_filepath = 'Best_Kaggle_Classification_Weights_{}_v4.h5'
cur_weights_path = train_weights_filepath.format('1024train')

# datagen params
full_train_params = get_train_params(batch_size, (1024, 1024), 1)
full_val_params = get_val_params(batch_size, (1024, 1024), 1)

# Create model
full_model = Inception_model(input_shape=(1024, 1024)+(n_channels,))
full_model.load_weights(cur_weights_path)

# Get datagen
_, full_val_gen, _ = get_class_datagen(
    train_pos_datapath, train_neg_datapath, full_train_params, full_val_params, val_split)

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
from prettytable import PrettyTable
table = PrettyTable(['Threshold', 'True Positive Rate', 'False Positive Rate'])
count = 0
for t,tp,fp in zip(thresholds,tpr,fpr):
    if count % 5 == 0:
        table.add_row(['{:.04f}'.format(t),'{:.04f}'.format(tp),'{:.04f}'.format(fp)])
    count += 1
print(table)


# Get and display a few predictions
for ind in range(5):
    b_ind = np.random.randint(0, len(full_val_gen))
    valX, valY = full_val_gen.__getitem__(b_ind)
    preds = full_model.predict_on_batch(valX)
    cur_im = valX[0]
    disp_im = np.concatenate([cur_im[..., c]
                              for c in range(cur_im.shape[-1])], axis=1)
    plt.imshow(disp_im, cmap='gray')
    plt.title('Predicted: {:.04f} Actual: {}'.format(preds[0,0], valY[0]))
    plt.show()