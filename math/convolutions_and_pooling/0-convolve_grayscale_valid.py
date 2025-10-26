#!/usr/bin/env python3
"""Valid grayscale image convolution without more than two for loops."""
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    Args:
        images (np.ndarray): shape (m, h, w) multiple grayscale images.
        kernel (np.ndarray): shape (kh, kw) convolution kernel.
    Returns:
        np.ndarray: convolved images (m, h-kh+1, w-kw+1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    conv = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = images[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return conv
