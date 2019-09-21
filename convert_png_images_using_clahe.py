#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:21:03 2019

@author: tjb129
"""
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
import os
from matplotlib import pyplot as plt

parent = '/data/Kaggle' 
folders = [ 'pos-all-png',
            'test-png',
            'train-png',
            'neg-filt-png',
            'pos-filt-png']
               
           

for dir_i in folders:
    dir_i = os.path.join(parent, dir_i)
    save_dir = dir_i + '-clahe'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for f in os.listdir(dir_i):
        fname = os.fsdecode(f)
        if fname.endswith(".png"):
            
            im = np.array(Image.open(os.path.join(dir_i, fname)))
            if len(im.shape) > 2:
                im = im[..., 0]
            im = im.astype(np.float)
            
            # normalize to [0,1]
            im /= 255.
            
            im = equalize_adapthist(im)
            
            im = np.uint8(im*255)
            
            im2 = Image.fromarray(im)
            im2.save(os.path.join(save_dir, fname))
            
            