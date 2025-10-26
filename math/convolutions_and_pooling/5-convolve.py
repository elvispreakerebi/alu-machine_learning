#!/usr/bin/env python3
"""Multi-channel, multi-kernel convolution, max 3 for loops (i, j, k)."""
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images with multiple kernels.

    Args:
        images (np.ndarray): (m, h, w, c) input images.
        kernels (np.ndarray): (kh, kw, c, nc) kernels.
        padding: ('valid', 'same', or (ph, pw)) padding mode.
        stride: (sh, sw) stride values.
    Returns:
        np.ndarray: convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:
        ph, pw = 0, 0
    images_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    h_pad, w_pad = images_pad.shape[1], images_pad.shape[2]
    h_out = (h_pad - kh) // sh + 1
    w_out = (w_pad - kw) // sw + 1
    output = np.zeros((m, h_out, w_out, nc))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(nc):
                hs, ws = i * sh, j * sw
                window = images_pad[:, hs:hs+kh, ws:ws+kw, :]
                output[:, i, j, k] = np.sum(window * kernels[:, :, :, k], axis=(1, 2, 3))
    return output
