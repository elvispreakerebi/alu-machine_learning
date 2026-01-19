#!/usr/bin/env python3
"""
Module for creating a layer in a neural network.
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer in a neural network.

    Args:
        prev (tf.Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation: The activation function that the layer should use.

    Returns:
        tf.Tensor: The tensor output of the layer.
    """
    # Initialize weights using He et al. initialization
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    # Create the dense layer
    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    
    return layer
