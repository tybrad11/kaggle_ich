import csv
import datetime
import os
from glob import glob
from os.path import join

import cv2
import GPUtil
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from natsort import natsorted
from PIL import Image
from scipy.ndimage.measurements import label as scipy_label
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Datagen import PngDataGenerator, PngClassDataGenerator
from Losses import dice_coef_loss
from mask_functions_pneumothorax import mask2rle
from Models import BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED
from ProcessMasks import CleanMask_v1


class PneumothoraxModel:
    def __init__(self,
                 test_path='/data/Kaggle/test-png',
                 dims=(512, 512),
                 batch_size=8,
                 val_split=.2,
                 epochs=1,
                 optimizer='Adam',
                 loss='dice',
                 multi_process=True):
        try:
            if not 'DEVICE_ID' in locals():
                DEVICE_ID = GPUtil.getFirstAvailable()[0]
                print('Using GPU', DEVICE_ID)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        except Exception as e:
            raise('No GPU available')
        self.test_path = test_path
        self.dims = dims
        self.n_channels = 1
        self.batch_size = batch_size
        self.val_split = val_split
        self.epochs = epochs
        self.multi_process = multi_process
        self.rng = np.random.RandomState()
        self.optimzer_str = optimizer
        self.loss_str = loss
        self.init_functions()

    def init_functions(self):
        self.init_aug_params()
        self.init_optimization()
        self.init_callbacks()

    def init_aug_params(self):
        self.train_aug_params = {'batch_size': self.batch_size,
                                 'dim': self.dims,
                                 'n_channels': 1,
                                 'shuffle': True,
                                 'rotation_range': 10,
                                 'width_shift_range': 0.1,
                                 'height_shift_range': 0.1,
                                 'brightness_range': None,
                                 'shear_range': 0.,
                                 'zoom_range': 0.1,
                                 'channel_shift_range': 0.,
                                 'fill_mode': 'constant',
                                 'cval': 0.,
                                 'horizontal_flip': True,
                                 'vertical_flip': False,
                                 'rescale': None,
                                 'preprocessing_function': None,
                                 'interpolation_order': 1}
        self.val_aug_params = {'batch_size': batch_size,
                               'dim': dims,
                               'n_channels': 1,
                               'shuffle': True,
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
                               'preprocessing_function': None,
                               'interpolation_order': 1}

    
    def init_optimization(self, lr=1e-4):
        if self.optimzer_str.lower() == 'adam':
            self.optimizer = keras.optimizers.Adam(lr=lr)
        else:
            raise('Unknown optimizer')
        if self.loss_str.lower() == 'dice':
            self.loss = dice_coef_loss
        else:
            raise('Unknown loss function')

    def init_CED_model(self, weights=None):
        print('Initializing model...')
        self.model = self._get_model()
        if weights is not None:
            self.model.load_weights(weights)
        self.model.compile(self.optimizer, loss=self.loss)

    def init_Class_model(self, weights=None):
        print('Initializing classification model...')
        self.class_model = self._get_class_model()
        if weights is not None:
            self.class_model.load_weights(weights)
        self.model.compile(self.optimizer, loss='binary_crossentropy')

    def validation_split(self):
        print('Splitting data into train/validation...')
        trainX, valX, trainY, valY = train_test_split(
            self.img_files, self.mask_files, test_size=self.val_split, random_state=self.rng, shuffle=True)
        train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
        val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])
        self.trainX = trainX
        self.valX = valX
        self.trainY = train_dict
        self.valY = valY        

    def setup_datagen(self,image_path,labels):
        print('Setting up data generators...')
        self.img_files = natsorted(glob(join(self.image_path, '*.png')))
        if isinstance(labels,str):
            self.labels = natsorted(glob(join(self.mask_path, '*.png')))
            self.is_classification=False
            try:
                assert len(self.img_files) == len(self.labels)
            except AssertionError:
                print("WARNING:")
                print('Number of image files and list files do not match')
        else:
            self.labels = labels
            self.is_classification=True

        self.train_gen = PngDataGenerator(self.trainX,
                                          self.trainY,
                                          **self.train_aug_params)
        self.val_gen = PngDataGenerator(self.valX,
                                        self.valY,
                                        **self.val_aug_params)
    def setup_class_datagen(self):
        print('Setting up classification data generators...')
        self.train_gen = PngClassDataGenerator(self.trainX,self.trainY,**self.train_aug_params)
        self.val_gen = PngClassDataGenerator(self.trainX,self.trainY,**self.val_aug_params)

    def _get_model(self, filt_num=16, numBlocks=4):
        return BlockModel2D(input_shape=self.dims+(self.n_channels,),
                            filt_num=filt_num, numBlocks=numBlocks)

    def init_callbacks(self,
                       filename=None,
                       use_checkpoint=True,
                       use_plateau=True,
                       use_earlystop=True):
        if filename is None:
            if self.multi_process:
                self.cur_weights_name = 'Pneumothorax_model_weights_{epoch:02d}-{val_loss:.4f}.h5'
            else:
                timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H")
                self.cur_weights_name = 'Best_Model_Weights_dim{}_date{}.h5'.format(
                    self.dims[0], timestamp)

        callbacks = []
        if use_checkpoint:
            callbacks.append(ModelCheckpoint(self.cur_weights_name, monitor='val_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1))
        if use_plateau:
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', factor=.5, patience=3, verbose=1))
        if use_earlystop:
            callbacks.append(EarlyStopping(monitor='val_loss',
                                           patience=5, restore_best_weights=True))
        self.callbacks = callbacks

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.epochs
        print('Beginning training...')
        self.history = self.model.fit_generator(generator=self.train_gen,
                                                epochs=epochs, use_multiprocessing=self.multi_process,
                                                workers=8, verbose=1, callbacks=self.callbacks,
                                                validation_data=self.val_gen)
        if self.multi_process:
            h5files = glob('*.h5')
            load_file = max(h5files, key=os.path.getctime)
        else:
            load_file = self.cur_weights_name
        print('Training complete. Loading best weights...')
        self.model.load_weights(load_file)
        print('Weights loaded.')

    def splitfile(self, file):
        _, file = os.path.split(file)
        return os.path.splitext(file)[0]

    def LoadImgForTest(self, f):
        img = Image.open(f)
        img = cv2.resize(np.array(img), self.dims).astype(np.float)
        img = self.normalize_image(img)
        return img

    def normalize_image(self, x):
        low_cut = np.percentile(x, 5)
        high_cut = np.percentile(x, 95)
        x -= low_cut
        x /= high_cut
        x[x < 0] = 0.
        x[x > 1] = 0.
        return x

    def generate_submission(self):
        test_datapath = '/data/Kaggle/test-png'
        # Get list of files
        img_files = natsorted(glob(join(test_datapath, '*.png')))
        # load files into array
        test_imgs = np.stack([self.LoadImgForTest(f, self.dims)
                              for f in img_files])[..., np.newaxis]
        # Get predicted masks
        tqdm.write('Getting mask predictions...')
        masks = self.model.predict(
            test_imgs, batch_size=self.batch_size, verbose=1)
        # data to write to csv
        submission_data = []
        # process mask
        for ind, cur_file in enumerate(tqdm(img_files)):
            cur_mask = masks[ind, ..., 0]
            cur_im = test_imgs[ind, ..., 0]
            cur_mask = (cur_mask > .5).astype(np.int)
            cur_id = splitfile(cur_file)

            processed_mask = CleanMask_v1(cur_mask)
            lbl_mask, numObj = scipy_label(processed_mask)
            if numObj > 0:
                for label in range(1, numObj+1):
                    temp_mask = np.zeros_like(cur_mask)
                    temp_mask[lbl_mask == label] = 1
                    temp_mask = cv2.resize(
                        temp_mask.astype(np.float), (1024, 1024))
                    temp_mask[temp_mask < .5] = 0
                    temp_mask[temp_mask > 0] = 255
                    temp_mask = np.transpose(temp_mask)
                    cur_rle = mask2rle(temp_mask, 1024, 1024)
                    submission_data.append([cur_id, cur_rle])
            else:
                cur_rle = -1
                submission_data.append([cur_id, cur_rle])

        # write to csv
        # generate time-stamped filename
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
        csv_filename = 'TestSubmission_{}'.format(timestamp)
        tqdm.write('Writing csv...')
        with open(csv_filename, mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            for row in tqdm(submission_data):
                writer.writerow(row)

        print('Done')
