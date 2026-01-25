#!/usr/bin/env python3
"""
Module for creating batch normalization layers in TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation: The activation function that should be used on the output
                   of the layer.

    Returns:
        tf.Tensor: A tensor of the activated output for the layer.
    """
    # Create the Dense layer with variance scaling initializer
    # Use use_bias=False since we'll use beta for bias
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=kernel_initializer,
        use_bias=False
    )
    
    # Get the unactivated output Z from the Dense layer
    Z = dense_layer(prev)
    
    # Create trainable parameters gamma and beta
    # gamma: initialized to 1, shape (n,)
    # beta: initialized to 0, shape (n,)
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    
    # Calculate mean and variance along axis 0 (across batch dimension)
    # Keep dimensions for broadcasting: shape (1, n)
    mean, variance = tf.nn.moments(Z, axes=[0], keep_dims=True)
    
    # Normalize Z: (Z - mean) / sqrt(variance + epsilon)
    epsilon = 1e-8
    Z_norm = tf.divide(tf.subtract(Z, mean), 
                      tf.sqrt(tf.add(variance, epsilon)))
    
    # Scale and shift: gamma * Z_norm + beta
    # Reshape gamma and beta to (1, n) for broadcasting if needed
    gamma_reshaped = tf.reshape(gamma, [1, n])
    beta_reshaped = tf.reshape(beta, [1, n])
    Z_bn = tf.add(tf.multiply(Z_norm, gamma_reshaped), beta_reshaped)
    
    # Apply activation function
    activated_output = activation(Z_bn)
    
    return activated_output
