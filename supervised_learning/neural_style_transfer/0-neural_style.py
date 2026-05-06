#!/usr/bin/env python3
"""
Neural Style Transfer (NST).
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for neural style transfer.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor.

        Args:
            style_image (np.ndarray): Style reference image (h, w, 3).
            content_image (np.ndarray): Content reference image (h, w, 3).
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        if not tf.executing_eagerly():
            tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its pixels are between 0 and 1 and its largest
        side is 512 pixels.

        Args:
            image (np.ndarray): Image array of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor of shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        scale = 512 / max(h, w)
        h_new = int(np.round(h * scale))
        w_new = int(np.round(w * scale))

        img = tf.convert_to_tensor(image, dtype=tf.float32)
        if np.max(image) > 1:
            img = img / 255.0

        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_bicubic(img, size=(h_new, w_new))
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img
