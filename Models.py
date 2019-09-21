import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.initializers import RandomNormal
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Cropping2D, Dense, Flatten, GlobalAveragePooling2D,
                          Input, Lambda, MaxPooling2D, Reshape, UpSampling2D,
                          ZeroPadding2D, ZeroPadding3D, Add, concatenate)
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.models import Model

import keras_segmentation as keras_seg

# Parameterized 2D Block Model
def BlockModel2D(input_shape, filt_num=16, numBlocks=3):
    """Creates a Block CED model for segmentation problems
    Args:
        input shape: a list or tuple of [rows,cols,channels] of input images
        filt_num: the number of filters in the first and last layers
        This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        regression: Whether to have a continuous output with linear activation
    Returns:
        An unintialized Keras model

    Example useage: SegModel = BlockModel2D([256,256,1],filt_num=8)

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
    the rows/cols must be divisible by 2^numBlocks for skip connections
    to match up properly
    """
    use_bn = True

    # check for input shape compatibility
    rows, cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1, numBlocks+1)))/2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name='input_layer')

    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1, numBlocks+1):
        x1 = Conv2D(filt_num*rr, (1, 1), padding='same',
                    name='Conv1_{}'.format(rr))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*rr, (3, 3), padding='same',
                    name='Conv3_{}'.format(rr))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv51_{}'.format(rr))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv52_{}'.format(rr))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1, x3, x52], name='merge_{}'.format(rr))
        x = Conv2D(filt_num*rr, (1, 1), padding='valid',
                   name='ConvAll_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (4, 4), padding='valid',
                   strides=(2, 2), name='DownSample_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (3, 3), padding='same',
                   name='ConvClean_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_clean_{}'.format(rr))(x)
        skip_list.append(x)

    # expanding blocks
    expnums = list(range(1, numBlocks+1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd-1], x],
                            name='skip_connect_{}'.format(dd))
        x1 = Conv2D(filt_num*dd, (1, 1), padding='same',
                    name='DeConv1_{}'.format(dd))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_Dx1_{}'.format(dd))(x1)
        x3 = Conv2D(filt_num*dd, (3, 3), padding='same',
                    name='DeConv3_{}'.format(dd))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_Dx3_{}'.format(dd))(x3)
        x51 = Conv2D(filt_num*dd, (3, 3), padding='same',
                     name='DeConv51_{}'.format(dd))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_Dx51_{}'.format(dd))(x51)
        x52 = Conv2D(filt_num*dd, (3, 3), padding='same',
                     name='DeConv52_{}'.format(dd))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_Dx52_{}'.format(dd))(x52)
        x = concatenate([x1, x3, x52], name='Dmerge_{}'.format(dd))
        x = Conv2D(filt_num*dd, (1, 1), padding='valid',
                   name='DeConvAll_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dall_{}'.format(dd))(x)
        x = UpSampling2D(size=(2, 2), name='UpSample_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3), padding='same',
                   name='DeConvClean1_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean1_{}'.format(dd))(x)
        x = Conv2D(filt_num*dd, (3, 3), padding='same',
                   name='DeConvClean2_{}'.format(dd))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_Dclean2_{}'.format(dd))(x)

    # classifier
    lay_out = Conv2D(1, (1, 1), activation='sigmoid', name='output_layer')(x)

    return Model(lay_input, lay_out)

    # Parameterized 2D Block Model


def BlockModel_Classifier(input_shape, filt_num=16, numBlocks=3):
    """Creates a Block model for pretraining on classification task
    Args:
        input shape: a list or tuple of [rows,cols,channels] of input images
        filt_num: the number of filters in the first and last layers
        This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        regression: Whether to have a continuous output with linear activation
    Returns:
        An unintialized Keras model

    Example useage: SegModel = BlockModel2D([256,256,1],filt_num=8)

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
    the rows/cols must be divisible by 2^numBlocks for skip connections
    to match up properly
    """

    use_bn = True

    # check for input shape compatibility
    rows, cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1, numBlocks+1)))/2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name='input_layer')

    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1, numBlocks+1):
        x1 = Conv2D(filt_num*rr, (1, 1), padding='same',
                    name='Conv1_{}'.format(rr))(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*rr, (3, 3), padding='same',
                    name='Conv3_{}'.format(rr))(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv51_{}'.format(rr))(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*rr, (3, 3), padding='same',
                     name='Conv52_{}'.format(rr))(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1, x3, x52], name='merge_{}'.format(rr))
        x = Conv2D(filt_num*rr, (1, 1), padding='valid',
                   name='ConvAll_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (4, 4), padding='valid',
                   strides=(2, 2), name='DownSample_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*rr, (3, 3), padding='same',
                   name='ConvClean_{}'.format(rr))(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name='elu_skip_{}'.format(rr))(x)

    # average pooling
    x = GlobalAveragePooling2D()(x)
    # classifier
    lay_out = Dense(1, activation='sigmoid', name='output_layer')(x)

    return Model(lay_input, lay_out)


def ConvertEncoderToCED(model, model_name):
    # Returns a model with frozen encoder layers
    # and complimentary, unfrozen decoder layers
    # get input layer
    # model must be compiled again after using this function
    lay_input = model.input
    # get skip connection layer outputs
    skip_list = [l.output for l in model.layers if 'skip' in l.name]
    numBlocks = len(skip_list)
    filt_num = int(skip_list[0].shape[-1])
    x = model.layers[-3].output
    # freeze encoder layers
    for layer in model.layers:
        layer.trainable = False
    
    use_bn = True
    if model_name.lower() == 'block2d':
        # make expanding blocks
        expnums = list(range(1, numBlocks+1))
        expnums.reverse()
        for dd in expnums:
            if dd < len(skip_list):
                x = concatenate([skip_list[dd-1], x],
                                name='skip_connect_{}'.format(dd))
            x1 = Conv2D(filt_num*dd, (1, 1), padding='same',
                        name='DeConv1_{}'.format(dd))(x)
            if use_bn:
                x1 = BatchNormalization()(x1)
            x1 = ELU(name='elu_Dx1_{}'.format(dd))(x1)
            x3 = Conv2D(filt_num*dd, (3, 3), padding='same',
                        name='DeConv3_{}'.format(dd))(x)
            if use_bn:
                x3 = BatchNormalization()(x3)
            x3 = ELU(name='elu_Dx3_{}'.format(dd))(x3)
            x51 = Conv2D(filt_num*dd, (3, 3), padding='same',
                         name='DeConv51_{}'.format(dd))(x)
            if use_bn:
                x51 = BatchNormalization()(x51)
            x51 = ELU(name='elu_Dx51_{}'.format(dd))(x51)
            x52 = Conv2D(filt_num*dd, (3, 3), padding='same',
                         name='DeConv52_{}'.format(dd))(x51)
            if use_bn:
                x52 = BatchNormalization()(x52)
            x52 = ELU(name='elu_Dx52_{}'.format(dd))(x52)
            x = concatenate([x1, x3, x52], name='Dmerge_{}'.format(dd))
            x = Conv2D(filt_num*dd, (1, 1), padding='valid',
                       name='DeConvAll_{}'.format(dd))(x)
            if use_bn:
                x = BatchNormalization()(x)
            x = ELU(name='elu_Dall_{}'.format(dd))(x)
            x = UpSampling2D(size=(2, 2), name='UpSample_{}'.format(dd))(x)
            x = Conv2D(filt_num*dd, (3, 3), padding='same',
                       name='DeConvClean1_{}'.format(dd))(x)
            if use_bn:
                x = BatchNormalization()(x)
            x = ELU(name='elu_Dclean1_{}'.format(dd))(x)
            x = Conv2D(filt_num*dd, (3, 3), padding='same',
                       name='DeConvClean2_{}'.format(dd))(x)
            if use_bn:
                x = BatchNormalization()(x)
            x = ELU(name='elu_Dclean2_{}'.format(dd))(x)
    
        # classifier
        lay_out = Conv2D(1, (1, 1), activation='sigmoid', name='output_layer')(x)
        
        
    elif  model_name.lower() == 'resunet':
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip_list[2]])
        res5 = Conv2D(96, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
        x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x,res5])
        
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip_list[1]])
        res6 = Conv2D(48, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
        x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x,res6])
        
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip_list[0]])
        res7 = Conv2D(24, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
        x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x,res7])
        
        # end part
        x = Conv2D(1, (1,1), activation='relu', padding='same')(x)
        res_input = Conv2D(1, (1,1), activation='relu', padding='same')(lay_input)#this seems likely the step, although the graph isn’t clear on this
        x = Add()([x, res_input])
        
        lay_out = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)
        
        
        
        
    return Model(lay_input, lay_out)


def Inception_model(input_shape=(299, 299, 3)):
    incep_model = InceptionV3(
        include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    input_layer = incep_model.input
    incep_output = incep_model.output
    # x = Conv2D(16, (3, 3), activation='relu')(incep_output)
    # x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(incep_output)
    return Model(inputs=input_layer, outputs=x)

def densenet_model(input_shape=(224, 224, 3)):
    class_model = DenseNet121(
        include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    input_layer = class_model.input
    incep_output = class_model.output
    # x = Conv2D(16, (3, 3), activation='relu')(incep_output)
    # x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(incep_output)
    return Model(inputs=input_layer, outputs=x)


from efficientnet.keras import EfficientNetB0, EfficientNetB4, EfficientNetB7
def efficient_model(input_shape=(224,224,3)):
    model= EfficientNetB0(input_shape=input_shape, weights=None) #can do B7 if want larger
    input_layer = model.input
    last_layer = model.layers[-2].output
    x = Dense(1, activation='sigmoid')(last_layer)
    return Model(inputs=input_layer, outputs=x)


def res_unet(input_shape):
    
    input_tensor = Input(shape=(input_shape),name='input_layer')
    x = Conv2D(24, (3,3), activation='relu', padding='same')(input_tensor)
    res1 = Conv2D(24, (1,1), activation='relu', padding='same')(input_tensor) #to be added at end of block1
    x = BatchNormalization()(x)
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    block1 = Add()([x,res1]) #to be contatenated with decoder
    
    x = MaxPooling2D((2, 2))(block1)
    res2 = Conv2D(48, (1,1), activation='relu', padding='same')(x)#to be added at end of block2
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    block2 = Add()([x, res2])#to be contatenated with decoder
    
    x = MaxPooling2D((2, 2))(block2)
    res3 = Conv2D(96, (1,1), activation='relu', padding='same')(x)#to be added at end of block3
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    block3 = Add()([x, res3])#to be contatenated with decoder
    
    x = MaxPooling2D((2, 2))(block3)
    res4 = Conv2D(192, (1,1), activation='relu', padding='same')(x)#to be added at end of block4
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, res4])# now go to Decoder side -> UpSampling
    
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, block3])
    res5 = Conv2D(96, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,res5])
    
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, block2])
    res6 = Conv2D(48, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,res6])
    
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, block1])
    res7 = Conv2D(24, (1,1), activation='relu', padding='same')(x)#to be added at the end of block
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,res7])
    
    # end part
    x = Conv2D(1, (1,1), activation='relu', padding='same')(x)
    res_input = Conv2D(1, (1,1), activation='relu', padding='same')(input_tensor)#this seems likely the step, although the graph isn’t clear on this
    x = Add()([x, res_input])
    
    out = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)
    
    return Model(inputs=input_tensor, outputs=out)




def res_unet_encoder(input_shape):
    
    input_tensor = Input(shape=(input_shape),name='input_layer')
    x = Conv2D(24, (3,3), activation='relu', padding='same')(input_tensor)
    res1 = Conv2D(24, (1,1), activation='relu', padding='same')(input_tensor) #to be added at end of block1
    x = BatchNormalization()(x)
    x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x,res1]) #to be contatenated with decoder
    block1 = Conv2D(24, (3,3), activation='relu', padding='same', name='skip_1')(x)
    
    x = MaxPooling2D((2, 2))(block1)
    res2 = Conv2D(48, (1,1), activation='relu', padding='same')(x)#to be added at end of block2
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, res2])#to be contatenated with decoder
    block2 = Conv2D(48, (3,3), activation='relu', padding='same', name='skip_2')(x)
    
    x = MaxPooling2D((2, 2))(block2)
    res3 = Conv2D(96, (1,1), activation='relu', padding='same')(x)#to be added at end of block3
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, res3])#to be contatenated with decoder
    block3 = Conv2D(96, (3,3), activation='relu', padding='same', name='skip_3')(x)
    
    x = MaxPooling2D((2, 2))(block3)
    res4 = Conv2D(192, (1,1), activation='relu', padding='same')(x)#to be added at end of block4
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, res4])
    
    #this is where we'd normally go to the decoder...but we'll make it a classificatino model
    x = Conv2D(192, (3,3), activation='relu', padding='same')(x)    
    x = GlobalAveragePooling2D()(x)
    # classifier
    lay_out = Dense(1, activation='sigmoid', name='output_layer')(x)
    
    return Model(inputs=input_tensor, outputs=lay_out)




  #############################################################################
 ######################### borrowed from internet ############################
#############################################################################

  #######################    
 #### Attention U-net ##
#######################

import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,BatchNormalization,concatenate,Activation
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU



def _MiniUnet(input,shape):
    x1 = Conv2D(shape, (3, 3), strides=(1, 1), padding="same",activation="relu")(input)
    pool1=MaxPooling2D(pool_size=(2, 2))(x1)
    
    x2 = Conv2D(shape*2, (3, 3), strides=(1, 1), padding="same",activation="relu")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    
    x3 = Conv2D(shape * 3, (3, 3), strides=(1, 1), padding="same",activation="relu")(pool2)
    
    x=concatenate([UpSampling2D(size=(2,2))(x3),x2],axis=3)
    x = Conv2D(shape*2, (3, 3), strides=(1, 1), padding="same",activation="relu")(x)
    
    x = concatenate([UpSampling2D(size=(2, 2))(x),x1],axis=3)
    x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same", activation="sigmoid")(x)
    return x

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16
    
    theta_x = Conv2D(inter_shape, (3, 3), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16
    
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    
    # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
    # upsample_psi=my_repeat([upsample_psi])
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    
    y = multiply([upsample_psi, x])
    
    # print(K.is_keras_tensor(upsample_psi))
    
    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def UnetGatingSignal(input, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def UnetConv2D(input, outdim, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x =BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def UnetConv2DPro(input, outdim):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    attn_shortcut=_MiniUnet(input,outdim)
    
    merge=multiply([attn_shortcut,x])
    result=add([merge,x])
    return result


def attention_unet(input_shape):
    inputs = Input((input_shape))
    conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
    conv = LeakyReLU(alpha=0.3)(conv)
    
    conv1 = UnetConv2D(conv, 32,is_batchnorm=True)  # 32 128
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 32,is_batchnorm=True)  # 32 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = UnetConv2D(pool2, 64,is_batchnorm=True)  # 64 32
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UnetConv2D(pool3, 64,is_batchnorm=True)  # 64 16
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    center = UnetConv2D(pool4, 128,is_batchnorm=True)  # 128 8
    
    gating = UnetGatingSignal(center, is_batchnorm=True)
    attn_1 = AttnGatingBlock(conv4, gating, 128)
    up1 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(center), attn_1], axis=3)
    
    gating = UnetGatingSignal(up1, is_batchnorm=True)
    attn_2 = AttnGatingBlock(conv3, gating, 64)
    up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(up1), attn_2], axis=3)
    
    gating = UnetGatingSignal(up2, is_batchnorm=True)
    attn_3 = AttnGatingBlock(conv2, gating, 32)
    up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up2), attn_3], axis=3)
    
    up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up3), conv1], axis=3)
    
    
    conv8 = Conv2D(1, (1, 1), activation='relu', padding='same')(up4)
#    conv8 = core.Reshape((input_shape[0] * input_shape[1],(1)))(conv8)
    ############
    act = Activation('sigmoid')(conv8)
    
    return Model(inputs=inputs, outputs=act)

   ###############################################     
  ### END OF   -- Attention U-net ###############
 ############################################### 



  #######################    
 #### tiramisu net #####
#######################

import keras.models as models
from keras.layers.core import Permute
import json

# weight_decay = 0.0001
from keras.regularizers import l2



def DenseBlock(layers_count, filters, previous_layer, model_layers, level):
    model_layers[level] = {}
    for i in range(layers_count):
        model_layers[level]['b_norm'+str(i+1)] = BatchNormalization(mode=0, axis=3,
                                     gamma_regularizer=l2(0.0001),
                                     beta_regularizer=l2(0.0001))(previous_layer)
        model_layers[level]['act'+str(i+1)] = Activation('relu')(model_layers[level]['b_norm'+str(i+1)])
        model_layers[level]['conv'+str(i+1)] = Conv2D(filters,   kernel_size=(3, 3), padding='same',
                                    kernel_initializer="he_uniform",
                                    data_format='channels_last')(model_layers[level]['act'+str(i+1)])
        model_layers[level]['drop_out'+str(i+1)] = Dropout(0.2)(model_layers[level]['conv'+str(i+1)])
        previous_layer  = model_layers[level]['drop_out'+str(i+1)]
    # print(model_layers)
    return model_layers[level]['drop_out'+ str(layers_count)] # return last layer of this level 


def TransitionDown(filters, previous_layer, model_layers, level):
    model_layers[level] = {}
    model_layers[level]['b_norm'] = BatchNormalization(mode=0, axis=3,
                                 gamma_regularizer=l2(0.0001),
                                 beta_regularizer=l2(0.0001))(previous_layer)
    model_layers[level]['act'] = Activation('relu')(model_layers[level]['b_norm'])
    model_layers[level]['conv'] = Conv2D(filters, kernel_size=(1, 1), padding='same',
                              kernel_initializer="he_uniform")(model_layers[level]['act'])
    model_layers[level]['drop_out'] = Dropout(0.2)(model_layers[level]['conv'])
    model_layers[level]['max_pool'] = MaxPooling2D(pool_size=(2, 2),
                            strides=(2, 2),
                            data_format='channels_last')(model_layers[level]['drop_out'])
    return model_layers[level]['max_pool']

def TransitionUp(filters,input_shape,output_shape, previous_layer, model_layers, level):
    model_layers[level] = {}
    model_layers[level]['conv'] = Conv2DTranspose(filters,  kernel_size=(3, 3), strides=(2, 2),
                                        padding='same',
                                        output_shape=output_shape,
                                        input_shape=input_shape,
                                        kernel_initializer="he_uniform",
                                        data_format='channels_last')(previous_layer)

    return model_layers[level]['conv']

def tiramisu(input_shape):
    inputs = Input(input_shape)

    first_conv = Conv2D(48, kernel_size=(3, 3), padding='same', 
                            kernel_initializer="he_uniform",
                            kernel_regularizer = l2(0.0001),
                            data_format='channels_last')(inputs)
    # first 
    shape_1_up = (int(input_shape[0]/(2**5)), int(input_shape[1]/(2**5)))   
    shape_2_up = (int(input_shape[0]/(2**4)), int(input_shape[1]/(2**4)))
    shape_3_up = (int(input_shape[0]/(2**3)), int(input_shape[1]/(2**3)))
    shape_4_up = (int(input_shape[0]/(2**2)), int(input_shape[1]/(2**2)))
    shape_5_up = (int(input_shape[0]/(2**1)), int(input_shape[1]/(2**1)))
    
    f = [20, 30, 30, 40, 50, 60]
    
    enc_model_layers = {}

    layer_1_down  = DenseBlock(5,f[0], first_conv, enc_model_layers, 'layer_1_down' ) # 5*12 = 60 + 48 = 108
    layer_1a_down  = TransitionDown(f[0], layer_1_down, enc_model_layers,  'layer_1a_down')
    
    layer_2_down = DenseBlock(5,f[1], layer_1a_down, enc_model_layers, 'layer_2_down' ) # 5*12 = 60 + 108 = 168
    layer_2a_down = TransitionDown(f[1], layer_2_down, enc_model_layers,  'layer_2a_down')
    
    layer_3_down  = DenseBlock(5,f[2], layer_2a_down, enc_model_layers, 'layer_3_down' ) # 5*12 = 60 + 168 = 228
    layer_3a_down  = TransitionDown(f[2], layer_3_down, enc_model_layers,  'layer_3a_down')
    
    layer_4_down  = DenseBlock(5,f[3], layer_3a_down, enc_model_layers, 'layer_4_down' )# 5*12 = 60 + 228 = 288
    layer_4a_down  = TransitionDown(f[4], layer_4_down, enc_model_layers,  'layer_4a_down')
    
    layer_5_down  = DenseBlock(5,f[4], layer_4a_down, enc_model_layers, 'layer_5_down' ) # 5*12 = 60 + 288 = 348
    layer_5a_down  = TransitionDown(f[4], layer_5_down, enc_model_layers,  'layer_5a_down')

    layer_bottleneck  = DenseBlock(15,f[5], layer_5a_down, enc_model_layers,  'layer_bottleneck') # m = 348 + 5*12 = 408

    
    layer_1_up  = TransitionUp(f[4], (f[4],)+shape_1_up, (None,f[4],)+ shape_2_up, layer_bottleneck, enc_model_layers, 'layer_1_up')  # m = 348 + 5x12 + 5x12 = 468.
    skip_up_down_1 = concatenate([layer_1_up, enc_model_layers['layer_5_down']['conv'+ str(5)]], axis=-1)
    layer_1a_up  = DenseBlock(5,f[4], skip_up_down_1, enc_model_layers, 'layer_1a_up' )

    layer_2_up  = TransitionUp(f[3], (f[3],)+shape_2_up, (None,f[3],)+ shape_3_up, layer_1a_up, enc_model_layers, 'layer_2_up') # m = 288 + 5x12 + 5x12 = 408
    skip_up_down_2 = concatenate([layer_2_up, enc_model_layers['layer_4_down']['conv'+ str(5)]], axis=-1)
    layer_2a_up  = DenseBlock(5,f[3], skip_up_down_2, enc_model_layers, 'layer_2a_up' )

    layer_3_up  = TransitionUp(f[2],  (f[2],)+shape_3_up, (None,f[2],)+ shape_4_up, layer_2a_up, enc_model_layers, 'layer_3_up') # m = 228 + 5x12 + 5x12 = 348
    skip_up_down_3 = concatenate([layer_3_up, enc_model_layers['layer_3_down']['conv'+ str(5)]], axis=-1)
    layer_3a_up  = DenseBlock(5,f[2], skip_up_down_3, enc_model_layers, 'layer_3a_up' )

    layer_4_up  = TransitionUp(f[1],  (f[1],)+shape_4_up, (None,f[1],)+ shape_5_up, layer_3a_up, enc_model_layers, 'layer_4_up') # m = 168 + 5x12 + 5x12 = 288
    skip_up_down_4 = concatenate([layer_4_up, enc_model_layers['layer_2_down']['conv'+ str(5)]], axis=-1)
    layer_4a_up  = DenseBlock(5,f[1], skip_up_down_4, enc_model_layers, 'layer_4a_up' )

    layer_5_up  = TransitionUp(f[0],  (f[0],)+shape_5_up, (None, f[0], input_shape[1], input_shape[1]), layer_4a_up, enc_model_layers, 'layer_5_up') # m = 108 + 5x12 + 5x12 = 228
    skip_up_down_5 = concatenate([layer_5_up, enc_model_layers['layer_1_down']['conv'+ str(5)]], axis=-1)
    layer_5a_up  = DenseBlock(5,f[0], skip_up_down_5, enc_model_layers, 'concatenate' )

    # last 
    last_conv = Conv2D(1, activation='sigmoid',
                            kernel_size=(1,1), 
                            padding='same',
                            kernel_regularizer = l2(0.0001),
                            data_format='channels_last')(layer_5a_up)
        
    
    
    return Model(inputs=[inputs], outputs=[last_conv])





   ###############################################     
  ### END OF   -- tiramisu net ##################
 ############################################### 
