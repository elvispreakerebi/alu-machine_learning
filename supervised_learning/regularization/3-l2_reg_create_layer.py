#!/usr/bin/env python3
"""
Module for creating TensorFlow layers with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization.

    Args:
        prev (tf.Tensor): Tensor containing the output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation: The activation function that should be used on the layer.
                   Can be None for no activation.
        lambtha (float): The L2 regularization parameter.

    Returns:
        tf.Tensor: The output of the new layer.
    """
    # Initialize weights using variance scaling initializer
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    # Create L2 regularizer
    kernel_regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    
    # Create the dense layer with L2 regularization
    output = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name='layer'
    )
    
    return output
