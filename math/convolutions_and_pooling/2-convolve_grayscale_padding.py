#!/usr/bin/env python3
"""Convolution with custom padding for grayscale images, two loops only."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom zero padding.

    Args:
        images (np.ndarray): m, h, w input images
        kernel (np.ndarray): kh, kw convolution kernel
        padding (tuple): (ph, pw) padding in height/width
    Returns:
        np.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    h_out = h + 2 * ph - kh + 1
    w_out = w + 2 * pw - kw + 1
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    conv = np.zeros((m, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            window = images_padded[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return conv
