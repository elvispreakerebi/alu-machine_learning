#!/usr/bin/env python3
"""
Module for creating the forward propagation graph for a neural network.
"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.Tensor): The placeholder for the input data.
        layer_sizes (list): A list containing the number of nodes in each layer
                           of the network.
        activations (list): A list containing the activation functions for each
                          layer of the network.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """
    # Start with the input
    prev = x
    
    # Create each layer sequentially
    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i] if i < len(activations) else None
        prev = create_layer(prev, n, activation)
    
    return prev
