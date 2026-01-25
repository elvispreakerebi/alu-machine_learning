#!/usr/bin/env python3
"""
Module for forward propagation with dropout.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X (numpy.ndarray): Array of shape (nx, m) containing the input data
                          for the network, where nx is the number of input
                          features and m is the number of data points.
        weights (dict): Dictionary of the weights and biases of the neural network.
        L (int): The number of layers in the network.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        dict: Dictionary containing:
            - A0, A1, A2, ..., AL: Outputs of each layer
            - D1, D2, ..., D(L-1): Dropout masks used on each hidden layer
    """
    cache = {}
    
    # Store input as A0
    cache['A0'] = X
    A = X
    
    # Forward propagation through hidden layers (1 to L-1)
    for l in range(1, L):
        W = weights['W{}'.format(l)]
        b = weights['b{}'.format(l)]
        
        # Compute linear transformation
        Z = np.dot(W, A) + b
        
        # Apply tanh activation
        A = np.tanh(Z)
        
        # Create dropout mask: random binary matrix with probability keep_prob
        # Shape: same as A (number of nodes in layer l, m)
        D = np.random.binomial(1, keep_prob, size=A.shape)
        
        # Apply dropout: scale by 1/keep_prob to maintain expected value
        A = A * D / keep_prob
        
        # Store activation and dropout mask
        cache['A{}'.format(l)] = A
        cache['D{}'.format(l)] = D
    
    # Forward propagation through output layer (L)
    W = weights['W{}'.format(L)]
    b = weights['b{}'.format(L)]
    
    # Compute linear transformation
    Z = np.dot(W, A) + b
    
    # Apply softmax activation
    # For numerical stability, subtract max before exponentiating
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    # Store output layer activation (no dropout on output layer)
    cache['A{}'.format(L)] = A
    
    return cache
