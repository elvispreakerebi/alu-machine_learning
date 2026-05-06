#!/usr/bin/env python3
"""
Neural Style Transfer (NST).
"""

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


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

        After init, sets style_image, content_image, alpha, beta, model,
        gram_style_features, and content_feature.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (isinstance(alpha, bool) or
                not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (isinstance(beta, bool) or
                not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    def load_model(self):
        """
        Creates the model used to calculate cost.

        Builds a Keras model from VGG19 (no top) with the same input as VGG19
        and outputs the style layer activations followed by the content layer.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
        )
        vgg.trainable = False
        outputs = (
            [vgg.get_layer(name).output for name in self.style_layers] +
            [vgg.get_layer(self.content_layer).output]
        )
        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    def generate_features(self):
        """
        Extracts style Gram matrices and content layer activations.

        Sets gram_style_features from the style image's style-layer outputs
        and content_feature from the content image's content-layer output.
        """
        n_style = len(self.style_layers)
        style_outputs = self.model(self.style_image)
        self.gram_style_features = [
            self.gram_matrix(style_outputs[i]) for i in range(n_style)
        ]
        content_outputs = self.model(self.content_image)
        self.content_feature = content_outputs[-1]

    @staticmethod
    def gram_matrix(input_layer):
        """
        Computes the Gram matrix of a convolutional layer output.

        G_ij = (1 / (H * W)) * sum_{h,w} F_hwi * F_hwj

        Args:
            input_layer (tf.Tensor or tf.Variable): Shape (1, h, w, c).

        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c).
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        rank = input_layer.shape.ndims
        if rank is None or rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = tf.unstack(tf.shape(input_layer))
        feats = tf.reshape(input_layer, (1, h * w, c))
        gram = tf.matmul(feats, feats, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)
        return gram

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its largest side is 512 pixels, then
        rescales pixel values from [0, 255] to [0, 1] after bicubic resize.

        Args:
            image (np.ndarray): Image array of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor of shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h >= w:
            h_new = 512
            w_new = int(np.round(w * 512.0 / h))
        else:
            w_new = 512
            h_new = int(np.round(h * 512.0 / w))

        img = tf.convert_to_tensor(image, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_bicubic(img, size=(h_new, w_new))
        img = img / 255.0
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img
