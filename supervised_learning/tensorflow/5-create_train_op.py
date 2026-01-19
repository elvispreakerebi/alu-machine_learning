#!/usr/bin/env python3
"""
Module for creating the training operation for a neural network.
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss (tf.Tensor): The loss of the network's prediction.
        alpha (float): The learning rate.

    Returns:
        tf.Operation: An operation that trains the network using gradient descent.
    """
    # Create gradient descent optimizer with the given learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    
    # Create training operation that minimizes the loss
    train_op = optimizer.minimize(loss)
    
    return train_op
