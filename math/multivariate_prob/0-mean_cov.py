#!/usr/bin/env python3
"""Mean and covariance calculation for a dataset."""
import numpy as np

def mean_cov(X):
    """Calculates the mean and covariance of a dataset.

    Args:
        X (np.ndarray): n x d array, where n is the number of points and d the dimensions.

    Raises:
        TypeError: if X is not a 2D np.ndarray.
        ValueError: if n < 2.

    Returns:
        mean (np.ndarray): shape (1, d)
        cov (np.ndarray): shape (d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    # Center the data
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)
    return mean, cov
