#!/usr/bin/env python3
"""Convolution over (m, h, w, c) images with channel-wise kernel (two for loops max)."""
import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on color images with multi-channel kernels.

    Args:
        images (np.ndarray): (m, h, w, c) input images
        kernel (np.ndarray): (kh, kw, c) kernel
        padding: ('same', 'valid', or (ph, pw)) type/custom padding
        stride: (sh, sw), convolution stride
    Returns:
        np.ndarray: convolved images (m, h_out, w_out)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:  # 'valid'
        ph, pw = 0, 0
    images_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    h_pad, w_pad = images_pad.shape[1], images_pad.shape[2]
    h_out = (h_pad - kh) // sh + 1
    w_out = (w_pad - kw) // sw + 1
    conv = np.zeros((m, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            i_start, j_start = i * sh, j * sw
            window = images_pad[:, i_start:i_start+kh, j_start:j_start+kw, :]
            conv[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))
    return conv
