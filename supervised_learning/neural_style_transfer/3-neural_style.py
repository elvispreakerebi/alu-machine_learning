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
        style_features, gram_style_features, and content_feature.
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
        Max-pooling layers are replaced by average pooling (NST convention).
        Layer outputs are taken from the rebuilt forward path (not
        get_layer().output on reused layers, which can reference the old
        VGG graph).
        """
        style_names = list(self.style_layers)
        content_name = self.content_layer

        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
        )
        vgg.trainable = False
        x = vgg.input
        tensors_by_layer = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name,
                )(x)
            else:
                x = layer(x)
            tensors_by_layer[layer.name] = x
        out_tensors = (
            [tensors_by_layer[name] for name in style_names] +
            [tensors_by_layer[content_name]]
        )
        self.model = tf.keras.Model(inputs=vgg.input, outputs=out_tensors)

    def generate_features(self):
        """
        Extracts style Gram matrices and content layer activations.

        Sets style_features and gram_style_features from the style image's
        style-layer outputs and content_feature from the content image's
        content-layer output.
        Images are scaled to [0, 1] then passed through VGG preprocessing
        (ImageNet mean-centered BGR) before the network, matching VGG19.
        """
        n_style = len(self.style_layers)
        vgg19 = tf.keras.applications.vgg19
        style_pre = vgg19.preprocess_input(self.style_image * 255.0)
        content_pre = vgg19.preprocess_input(self.content_image * 255.0)
        style_out = self.model(style_pre)
        if not isinstance(style_out, (list, tuple)):
            style_list = [style_out]
        else:
            style_list = list(style_out)
        self.style_features = [style_list[i] for i in range(n_style)]
        self.gram_style_features = [
            self.gram_matrix(style_list[i]) for i in range(n_style)
        ]
        content_out = self.model(content_pre)
        if not isinstance(content_out, (list, tuple)):
            self.content_feature = content_out
        else:
            self.content_feature = content_out[-1]

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
