#!/usr/bin/env python3
"""
Module for creating a training operation using the RMSProp optimization algorithm.
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in TensorFlow using the
    RMSProp optimization algorithm.

    Args:
        loss (tf.Tensor): The loss of the network.
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight (decay rate).
        epsilon (float): A small number to avoid division by zero.

    Returns:
        tf.Operation: The RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
