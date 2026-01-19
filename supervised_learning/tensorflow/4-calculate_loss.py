#!/usr/bin/env python3
"""
Module for calculating the softmax cross-entropy loss of a prediction.
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y (tf.Tensor): A placeholder for the labels of the input data.
        y_pred (tf.Tensor): A tensor containing the network's predictions.

    Returns:
        tf.Tensor: A tensor containing the loss of the prediction.
    """
    # Calculate softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    
    return loss
