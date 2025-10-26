#!/usr/bin/env python3
"""Same convolution for grayscale images with two for loops max."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images with zero-padding.

    Args:
        images (np.ndarray): m, h, w input images (grayscale).
        kernel (np.ndarray): kh, kw kernel.
    Returns:
        np.ndarray: output of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2 if kh % 2 == 1 else kh // 2
    pad_w = (kw - 1) // 2 if kw % 2 == 1 else kw // 2
    images_padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )
    conv = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            window = images_padded[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return conv
