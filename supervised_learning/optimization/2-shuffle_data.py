#!/usr/bin/env python3
"""
Module for shuffling data points in two matrices.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): The first numpy array of shape (m, nx) to shuffle,
                          where m is the number of data points and nx is the
                          number of features in X.
        Y (numpy.ndarray): The second numpy array of shape (m, ny) to shuffle,
                          where m is the same number of data points as in X
                          and ny is the number of features in Y.

    Returns:
        tuple: A tuple containing:
            - X_shuffled (numpy.ndarray): The shuffled X matrix.
            - Y_shuffled (numpy.ndarray): The shuffled Y matrix.
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
