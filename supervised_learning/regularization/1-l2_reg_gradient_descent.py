#!/usr/bin/env python3
"""
Module for gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y (numpy.ndarray): One-hot numpy array of shape (classes, m) containing
                          the correct labels for the data.
        weights (dict): Dictionary of the weights and biases of the neural network.
        cache (dict): Dictionary of the outputs of each layer of the neural network.
        alpha (float): The learning rate.
        lambtha (float): The L2 regularization parameter.
        L (int): The number of layers of the network.

    Returns:
        None: Updates weights and biases in place.
    """
    m = Y.shape[1]
    
    # Get the output of the last layer (softmax activation)
    A_L = cache['A{}'.format(L)]
    
    # Output layer gradient (softmax with cross-entropy loss)
    dZ = A_L - Y
    
    # Backpropagate through all layers
    for l in range(L, 0, -1):
        # Get previous layer's activation
        A_prev = cache['A{}'.format(l - 1)]
        
        # Get current layer's weights and biases
        W = weights['W{}'.format(l)]
        b = weights['b{}'.format(l)]
        
        # Calculate gradients
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Add L2 regularization term to weight gradients
        dW = dW + (lambtha / m) * W
        
        # Update weights and biases in place
        weights['W{}'.format(l)] = W - alpha * dW
        weights['b{}'.format(l)] = b - alpha * db
        
        # Calculate dZ for previous layer (if not the first layer)
        if l > 1:
            A_prev_activated = cache['A{}'.format(l - 1)]
            # Tanh derivative: 1 - A^2
            dZ = np.dot(W.T, dZ) * (1 - A_prev_activated ** 2)
