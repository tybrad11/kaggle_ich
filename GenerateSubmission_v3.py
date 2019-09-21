import concurrent.futures
import csv
import math
import os
import sys
from datetime import timedelta
from glob import glob
from os.path import join
from time import time

import cv2
import GPUtil
import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from scipy.ndimage.measurements import label as scipy_label
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from HelperFunctions import ConvertModelOutputToLinear
from mask_functions_pneumothorax import mask2rle, rle2mask
from Models import BlockModel2D, Inception_model, res_unet
from ProcessMasks import CleanMask_v1
from VisTools import mask_viewer0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)

start_time = time()

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')


def splitfile(file):
    _, file = os.path.split(file)
    return os.path.splitext(file)[0]


# Testing images
test_datapath = '/data/Kaggle/test-png'

# Best weights directory



# Path to classification model weights to use
class_weight_dir = './SavedWeights/'
class_weights_fname = 'Classification_Inception_1024.h5'
class_weights_filepath = join(class_weight_dir, class_weights_fname)


# Path(s) to segmentation model weights to use
# Provide list for ensemble evaluation, string for single model
# seg_weight_name = ['Best_Kaggle_Weights_1024train.h5','Best_Kaggle_Weights_1024train_v2.h5','Best_Kaggle_Weights_1024train_v3.h5']
# seg_weight_filepath = [join(weight_dir,name) for name in seg_weight_fname]
seg_weight_dir = './'
seg_weight_fname1 = 'Best_Kaggle_Weights_block2d_1024train_v2.h5'
seg_weight_filepath1 = join(seg_weight_dir, seg_weight_fname1)

seg_weight_dir2 = './'
seg_weight_fname2 = 'Best_Kaggle_Weights_resunet_1024train_v3.h5'
seg_weight_filepath2 = join(seg_weight_dir2, seg_weight_fname2)

# Where to save submission output
submission_filepath = 'Submissions/Submission_v12.csv'

# Whether to use ensemble

# Whether to use CLAHE normalization in image pre-processing
use_clahe = True

# parameters
batch_size = 6
im_dims = (1024, 1024)
n_channels = 1
thresh = .85  # threshold for classification model
seg_thresh = 0.4

# Get list of testing files
img_files = natsorted(glob(join(test_datapath, '*.png')))

# Function for loading in testing images


def LoadImg(f, dims=(1024, 1024)):
    img = Image.open(f)
    img = cv2.resize(np.array(img), dims).astype(np.float)
    img /= 255.
    if use_clahe:
        img = equalize_adapthist(img)
    return img

# Function for generating submission data for a sample


def GetSubData(file, label, mask):
    mask = mask[..., 0]
    mask = (mask > seg_thresh).astype(np.int)
    fid = splitfile(file)

    if label == 0:
        return [fid, -1]

    processed_mask = CleanMask_v1(mask)
    lbl_mask, numObj = scipy_label(processed_mask)
    if numObj > 0:
        processed_mask[processed_mask > 0] = 255
        processed_mask = np.transpose(processed_mask)
        rle = mask2rle(processed_mask, 1024, 1024)
    else:
        rle = -1
    return [fid, rle]

# Function for getting linear output masks
# from segmentation model for ensemble purposes


def GetBlockModelMasks(weights_path, test_imgs, batch_size):
    # Create model
    tqdm.write('Loading segmentation model...')
    model = BlockModel2D(input_shape=im_dims+(n_channels,),
                         filt_num=16, numBlocks=4)
    # Load weights
    model.load_weights(weights_path)

    # convert to linear output layer- for better ensembling
    model = ConvertModelOutputToLinear(model)

    # Get predicted masks
    tqdm.write('Getting predicted masks...')
    masks = model.predict(test_imgs, batch_size=batch_size, verbose=0)
    del model
    return masks

# sigmoid function to apply after ensembling


def sigmoid(x):
    return 1 / (1 + math.e ** -x)


# load testing image files into array
tqdm.write('Loading images...')
img_list = list()
# using multi processing for efficiency
with concurrent.futures.ProcessPoolExecutor() as executor:
    for img_array in tqdm(executor.map(LoadImg, img_files), total=len(img_files)):
        # put results into correct output list
        img_list.append(img_array)
# convert into 4D stack for model evaluation
test_imgs = np.stack(img_list)[..., np.newaxis]


# Load classification model, for stratifying
tqdm.write('Loading classification model...')
class_model = Inception_model(input_shape=(1024, 1024)+(n_channels,))
class_model.load_weights(class_weights_filepath)

# Get classification predictions
tqdm.write('Making classification predictions...')
pred_labels = class_model.predict(test_imgs, batch_size=4, verbose=1)
pred_labels = (pred_labels[:, 0] > thresh).astype(np.int)

# remove model, to save memory
del class_model
tqdm.write('Finished with classification model')


# Get masks from segmentation model ensemble
#model 1
tqdm.write('Starting model ensemble...')
model1 = BlockModel2D(input_shape=im_dims+(n_channels,),
                     filt_num=16, numBlocks=4)
model1.load_weights(seg_weight_filepath1)
tqdm.write('Getting predicted masks, model1...')
masks1 = model1.predict(test_imgs, batch_size=batch_size, verbose=1)
del model1   
 
#model 2
if 'block' in seg_weight_filepath2.lower():
    model2 = BlockModel2D(input_shape=im_dims+(n_channels,),
                         filt_num=16, numBlocks=4)
elif 'resunet' in seg_weight_filepath2.lower():
    model2 = res_unet(input_shape=im_dims+(n_channels,))
        
model2.load_weights(seg_weight_filepath2)
tqdm.write('Getting predicted masks, model2...')
masks2 = model2.predict(test_imgs, batch_size=batch_size, verbose=1)
del model2

masks = (masks1 + masks2) / 2

del masks1, masks2


# data to write to csv
submission_data = []
# process mask
tqdm.write('Processing masks...')
with concurrent.futures.ProcessPoolExecutor() as executor:
    for sub_data in tqdm(executor.map(GetSubData, img_files, pred_labels, masks), total=len(img_files)):
        # put results into output list
        submission_data.append(sub_data)

# write to csv
tqdm.write('Writing csv...')
with open(submission_filepath, mode='w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['ImageId', 'EncodedPixels'])
    for data in submission_data:
        writer.writerow(data)

# write some images to png


def SaveImMaskAsPng(img, mask, name, sdir='.'):
    # make mask into rgba
    yellow_mask = np.repeat(mask, 4, axis=-1)
    yellow_mask[..., 2] = 0
    yellow_mask[..., 3] = .3*yellow_mask[..., 3]
    ymask = (255*yellow_mask).astype(np.uint8)
    # make background image into rgb and save
    bkgd = Image.fromarray((255*img).astype(np.uint8)).convert('RGB')
    im_name = '{}_image.png'.format(name)
    bkgd.save(join(sdir, im_name))
    # paste on mask image and save
    fgd = Image.fromarray(ymask)
    bkgd.paste(fgd, (0, 0), fgd)
    msk_name = '{}_w_mask.png'.format(name)
    bkgd.save(join(sdir, msk_name))


masks[masks > seg_thresh] = 1
masks[masks < seg_thresh] = 0
output_dir = 'SampleImagesAndMasks_v12'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
tqdm.write('Saving sample images and masks...')
n = 50
name = 'Sample_{}_{}'
for ind, img, mask, label in tqdm(zip(range(n), test_imgs[:n], masks[:n], pred_labels[:n]), total=n):
    if label:
        cur_name = name.format(ind, 'pos')
    else:
        cur_name = name.format(ind, 'neg')
    SaveImMaskAsPng(img[..., 0], mask, cur_name, output_dir)

print('Done')
finish_time = time()
print('Time elapsed: {}'.format(timedelta(seconds=finish_time-start_time)))
