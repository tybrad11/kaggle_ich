import keras.backend as K
import tensorflow as tf
#import numpy as np

def jac_met(y_true, y_pred):
    Xr = K.round(y_pred)
    X2r = K.round(y_true)
    intersection = Xr*X2r
    union = K.maximum(Xr, X2r)
    intsum = K.sum(intersection)
    unsum = K.sum(union)
    jacc =intsum/unsum
    return jacc

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    dc = dice_coef(y_true,y_pred)
    return 1 - dc