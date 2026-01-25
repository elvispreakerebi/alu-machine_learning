#!/usr/bin/env python3
"""
Module for normalizing a matrix.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
        X (numpy.ndarray): A numpy array of shape (d, nx) to normalize, where
                          d is the number of data points and nx is the number
                          of features.
        m (numpy.ndarray): A numpy array of shape (nx,) containing the mean of
                          all features of X.
        s (numpy.ndarray): A numpy array of shape (nx,) containing the standard
                          deviation of all features of X.

    Returns:
        numpy.ndarray: The normalized X matrix with shape (d, nx).
    """
    return (X - m) / s
