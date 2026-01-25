#!/usr/bin/env python3
"""
Module for batch normalization of neural network outputs.
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    Args:
        Z (numpy.ndarray): Array of shape (m, n) that should be normalized,
                          where m is the number of data points and n is the
                          number of features.
        gamma (numpy.ndarray): Array of shape (1, n) containing the scales
                              used for batch normalization.
        beta (numpy.ndarray): Array of shape (1, n) containing the offsets
                             used for batch normalization.
        epsilon (float): A small number used to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix of shape (m, n).
    """
    # Calculate mean along axis 0 (across data points, for each feature)
    # Shape: (n,)
    mean = np.mean(Z, axis=0, keepdims=True)
    
    # Calculate variance along axis 0
    # Shape: (n,)
    variance = np.var(Z, axis=0, keepdims=True)
    
    # Normalize Z: (Z - mean) / sqrt(variance + epsilon)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    
    # Scale and shift: gamma * Z_norm + beta
    Z_bn = gamma * Z_norm + beta
    
    return Z_bn
