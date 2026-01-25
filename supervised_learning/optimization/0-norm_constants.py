#!/usr/bin/env python3
"""
Module for calculating normalization constants.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): A numpy array of shape (m, nx) to normalize, where
                          m is the number of data points and nx is the number
                          of features.

    Returns:
        tuple: A tuple containing:
            - mean (numpy.ndarray): The mean of each feature with shape (nx,).
            - std (numpy.ndarray): The standard deviation of each feature with
              shape (nx,).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
