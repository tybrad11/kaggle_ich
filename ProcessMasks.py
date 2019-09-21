from scipy.ndimage import binary_fill_holes
from scipy.ndimage.measurements import label as scipy_label
import numpy as np
def CleanMask_v1(mask):
    # remove small objects and fill holes
    mask = (mask > .5).astype(np.int)
    mask = binary_fill_holes(mask)
    lbl_mask, numObj = scipy_label(mask)
    processed_mask = np.zeros_like(mask)
    minimum_cc_sum = .005*np.prod(mask.shape)
    for label in range(1, numObj+1):
        if np.sum(lbl_mask == label) > minimum_cc_sum:
            processed_mask[lbl_mask == label] = 1
    return processed_mask.astype(np.int)