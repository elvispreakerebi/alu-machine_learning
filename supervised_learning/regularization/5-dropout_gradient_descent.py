#!/usr/bin/env python3
"""
Module for gradient descent with dropout regularization.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent.

    Args:
        Y (numpy.ndarray): One-hot numpy array of shape (classes, m) containing
                          the correct labels for the data.
        weights (dict): Dictionary of the weights and biases of the neural network.
        cache (dict): Dictionary of the outputs and dropout masks of each layer.
        alpha (float): The learning rate.
        keep_prob (float): The probability that a node will be kept.
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
        
        # Update weights and biases in place
        weights['W{}'.format(l)] = W - alpha * dW
        weights['b{}'.format(l)] = b - alpha * db
        
        # Calculate dZ for previous layer (if not the first layer)
        if l > 1:
            # Get the activation of the previous layer (after dropout)
            A_prev_after_dropout = cache['A{}'.format(l - 1)]
            # Get the dropout mask for the previous layer
            D_prev = cache['D{}'.format(l - 1)]
            
            # Compute dA (gradient w.r.t. activation)
            dA = np.dot(W.T, dZ)
            
            # Apply dropout mask to gradients: only propagate through kept nodes
            # Scale by keep_prob to account for the scaling done in forward prop
            dA = dA * D_prev / keep_prob
            
            # Recover activation before dropout for tanh derivative
            # A_before = A_after * keep_prob / D
            # For D=0 nodes, A_before doesn't matter since dA=0
            # Use np.where to avoid division by zero
            A_prev_before_dropout = np.where(
                D_prev > 0,
                A_prev_after_dropout * keep_prob / D_prev,
                0
            )
            
            # Tanh derivative: 1 - A^2
            # Use the activation before dropout for the derivative
            dZ = dA * (1 - A_prev_before_dropout ** 2)
