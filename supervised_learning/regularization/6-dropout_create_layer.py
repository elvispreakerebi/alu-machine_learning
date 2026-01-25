#!/usr/bin/env python3
"""
Module for creating TensorFlow layers with dropout.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev (tf.Tensor): Tensor containing the output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation: The activation function that should be used on the layer.
                   Can be None for no activation.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        tf.Tensor: The output of the new layer.
    """
    # Initialize weights using variance scaling initializer
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    # Create the dense layer
    # If activation is None, we'll apply it separately after dropout
    # Otherwise, apply activation in the dense layer
    if activation is None:
        # No activation: create dense layer without activation
        layer_output = tf.layers.dense(
            inputs=prev,
            units=n,
            activation=None,
            kernel_initializer=kernel_initializer,
            name='layer'
        )
    else:
        # Has activation: create dense layer with activation
        layer_output = tf.layers.dense(
            inputs=prev,
            units=n,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name='layer'
        )
    
    # Apply dropout
    # tf.nn.dropout automatically scales by 1/keep_prob during training
    output = tf.nn.dropout(layer_output, keep_prob)
    
    return output
