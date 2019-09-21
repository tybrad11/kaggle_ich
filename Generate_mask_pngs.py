import concurrent.futures
import csv
import os
import sys
from glob import glob
from os.path import join
import shutil

import cv2
import GPUtil
import keras
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from scipy.ndimage.measurements import label as scipy_label
from tqdm import tqdm

train_datapath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'
output_dir = '/data/Kaggle/TrainingImagesAndMasks'

image_files = natsorted(glob(join(train_datapath,'*.png')))
mask_files = natsorted(glob(join(train_mask_path,'*.png')))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
tqdm.write('Copying sample images and masks...')


def GenerateImages(files):
    image_file = files[0]
    mask_file = files[1]
    mask_img = Image.open(mask_file)
    mask_array = np.array(mask_img).astype(np.float)
    is_pos = np.any(mask_array)
    if is_pos:
        img = Image.open(image_file).convert('RGB')
        yellow_mask = np.repeat(mask_array[...,np.newaxis],4,axis=-1)
        yellow_mask[...,2] = 0
        yellow_mask[...,3] = .3*yellow_mask[...,3]
        ymask = (yellow_mask).astype(np.uint8)
        fgd = Image.fromarray(ymask)
        img.paste(fgd,(0,0),fgd)
        img_name = os.path.split(image_file)[-1]
        msk_name = os.path.split(image_file)[-1][:-4] + '_mask.png'
        shutil.copyfile(image_file,join(output_dir,img_name))
        img.save(join(output_dir,msk_name))
    return is_pos
    
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for pos in tqdm(executor.map(GenerateImages, zip(image_files,mask_files)),total=len(image_files)):
        # put results into correct output list
        results.append(pos)

print('Found {} positive images'.format(sum(results)))