from PIL import Image
from VisTools import showmask
import csv
import glob
import os
import numpy as np
from tqdm import tqdm

import cv2
import pydicom as dcm
from natsort import index_natsorted, natsorted, order_by_index

from mask_functions_pneumothorax import rle2mask

train_datapath = '/data/Kaggle/train'
train_outpath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'
test_datapath = '/data/Kaggle/test'
test_outpath = '/data/Kaggle/test-png'
if not os.path.exists(train_outpath):
    os.mkdir(train_outpath)
if not os.path.exists(train_mask_path):
    os.mkdir(train_mask_path)
if not os.path.exists(test_outpath):
    os.mkdir(test_outpath)

csv_file = 'train-rle.csv'

def splitfile(file):
    _, file = os.path.split(file)
    return os.path.splitext(file)[0]


# Training images
all_files = [f for f in glob.glob(
    train_datapath + '**/**/*.dcm', recursive=True)]

# Convert dicoms to png
print('Converting train DICOMs to PNG...')
for cur_file in tqdm(all_files):
    ds = dcm.dcmread(cur_file)
    cur_name = splitfile(cur_file) + '.png'
    cur_outpath = os.path.join(train_outpath, cur_name)
    cv2.imwrite(cur_outpath, ds.pixel_array)

# Testing images
all_test_files = [f for f in glob.glob(
    test_datapath + '**/**/*.dcm', recursive=True)]

print('Converting train DICOMs to PNG...')
for cur_file in tqdm(all_test_files):
    ds = dcm.dcmread(cur_file)
    cur_name = splitfile(cur_file) + '.png'
    cur_outpath = os.path.join(test_outpath, cur_name)
    cv2.imwrite(cur_outpath, ds.pixel_array)


# Mask conversions

# Read csv file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    data = [row for row in reader]
data = data[1:]

# gather file ids
all_file_ids = [d[0] for d in data]
all_rle = [d[1] for d in data]
unq_file_ids = natsorted(list(set(all_file_ids)))

# Generate masks to PNG
print('Converting masks to PNG....')
for cur_id in tqdm(unq_file_ids):
    # find matches for current file id
    matches = [i for i, d in enumerate(all_file_ids) if d == cur_id]
    if len(all_rle[matches[0]]) < 4:
        # no annotation for this image
        # make blank mask
        mask = np.zeros((1024, 1024))
    elif len(matches) > 1:
        # combine multiple annotations into one mask
        cur_rle = [all_rle[m] for m in matches]
        masks = [rle2mask(r, 1024, 1024) for r in cur_rle]
        mask = sum(masks)
    else:
        # convert single annotation to mask
        cur_rle = all_rle[matches[0]]
        try:
            mask = rle2mask(cur_rle, 1024, 1024)
        except IndexError as e:
            print(cur_id)
            raise IndexError(e)
    # write mask to png
    mask = np.transpose(mask)
    cur_outpath = os.path.join(train_mask_path, cur_id+'.png')
    cv2.imwrite(cur_outpath, mask)

print('Done')


# Testing

# test_id = '1.2.276.0.7230010.3.1.4.8323329.3604.1517875178.653360'
test_id = unq_file_ids[21]

img = Image.open(os.path.join(train_outpath, test_id+'.png'))
mask_img = Image.open(os.path.join(train_mask_path, test_id+'.png'))
showmask(np.array(img), np.array(mask_img)/255)

# dcm_file = [f for f in all_files if test_id in f][0]
# test_img = dcm.dcmread(dcm_file).pixel_array
# rle_inds = [i for i,f in enumerate(all_file_ids) if test_id == f]
# test_rle = [all_rle[m] for m in rle_inds]
# masks = [rle2mask(r,1024,1024) for r in test_rle]
# mask = sum(masks)
# showmask(test_img,np.transpose(mask)/255)
# showmask(test_img,0*test_img)
