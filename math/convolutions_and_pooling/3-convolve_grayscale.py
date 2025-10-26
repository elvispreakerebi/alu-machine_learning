#!/usr/bin/env python3
"""
General grayscale image convolution with stride and padding.
At most two loops used (over output i, j).
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with stridevarious paddings.

    Args:
        images (np.ndarray): (m, h, w) input images
        kernel (np.ndarray): (kh, kw) kernel
        padding: ('same', 'valid', or (ph, pw)), type of or custom padding
        stride: (sh, sw), convolution strides
    Returns:
        np.ndarray: output of the convolution
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:  # 'valid'
        ph, pw = 0, 0
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    h_padded = images_padded.shape[1]
    w_padded = images_padded.shape[2]
    h_out = (h_padded - kh) // sh + 1
    w_out = (w_padded - kw) // sw + 1
    conv = np.zeros((m, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            start_i = i * sh
            start_j = j * sw
            window = images_padded[:, start_i:start_i+kh, start_j:start_j+kw]
            conv[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return conv
