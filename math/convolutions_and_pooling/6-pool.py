#!/usr/bin/env python3
"""
Pooling on multi-channel images with max or average.
Only 2 for loops used (over spatial output indices).
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images using the specified mode and kernel/stride.

    Args:
        images (np.ndarray): (m, h, w, c) images
        kernel_shape (tuple): (kh, kw), the pooling window size
        stride (tuple): (sh, sw), stride along height and width
        mode (str): 'max' or 'avg' for pooling type
    Returns:
        np.ndarray: pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1
    pooled = np.zeros((m, h_out, w_out, c))
    for i in range(h_out):
        for j in range(w_out):
            i_start, j_start = i * sh, j * sw
            window = images[:, i_start:i_start+kh, j_start:j_start+kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.mean(window, axis=(1, 2))
    return pooled
