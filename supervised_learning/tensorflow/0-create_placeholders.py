#!/usr/bin/env python3
"""
Module for creating TensorFlow placeholders.
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for the neural network.

    Args:
        nx (int): The number of feature columns in the data.
        classes (int): The number of classes in the classifier.

    Returns:
        tuple: A tuple containing:
            - x (tf.Tensor): Placeholder for the input data to the neural network
              with shape (None, nx) and dtype float32.
            - y (tf.Tensor): Placeholder for the one-hot labels for the input data
              with shape (None, classes) and dtype float32.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
