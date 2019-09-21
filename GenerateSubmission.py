import csv
import os
from glob import glob
from os.path import join

import cv2
import GPUtil
import keras
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from scipy.ndimage.measurements import label as scipy_label
from tqdm import tqdm

from mask_functions_pneumothorax import mask2rle, rle2mask
from Models import BlockModel2D
from ProcessMasks import CleanMask_v1

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


test_datapath = '/data/Kaggle/test-norm-png-V2'
weight_filepath = 'Best_Kaggle_Weights_1024train.h5'
submission_filepath = 'Submission_v2.csv'

# parameters
batch_size = 4
im_dims = (1024, 1024)
n_channels = 1

# Get list of files
img_files = natsorted(glob(join(test_datapath, '*.png')))


def LoadImg(f, dims):
    img = Image.open(f)
    img = cv2.resize(np.array(img), dims).astype(np.float)
    img /= 255.
    return img


# load files into array
tqdm.write('Loading images...')
test_imgs = np.stack([LoadImg(f, im_dims)
                      for f in tqdm(img_files)])[..., np.newaxis]

# Create model
model = BlockModel2D(input_shape=im_dims+(n_channels,),
                     filt_num=16, numBlocks=4)
# Load weights
model.load_weights(weight_filepath)

# Get predicted masks
tqdm.write('Getting predicted masks...')
masks = model.predict(test_imgs, batch_size=batch_size, verbose=1)

# data to write to csv
submission_data = []
# process mask
tqdm.write('Processing masks...')
for ind, cur_file in enumerate(tqdm(img_files)):
    cur_mask = masks[ind, ..., 0]
    cur_im = test_imgs[ind, ..., 0]
    cur_mask = (cur_mask > .5).astype(np.int)
    cur_id = splitfile(cur_file)

    processed_mask = CleanMask_v1(cur_mask)
    lbl_mask, numObj = scipy_label(processed_mask)
    if numObj > 0:
        processed_mask[processed_mask > 0] = 255
        processed_mask = np.transpose(processed_mask)
        cur_rle = mask2rle(processed_mask, 1024, 1024)
    else:
        cur_rle = -1
    submission_data.append([cur_id, cur_rle])

# write to csv
tqdm.write('Writing csv...')
with open(submission_filepath, mode='w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['ImageId', 'EncodedPixels'])
    for data in tqdm(submission_data):
        writer.writerow(data)

# display some images
from VisTools import mask_viewer0
mask_viewer0(test_imgs[:100,...,0],masks[:100,...,0])

print('Done')
