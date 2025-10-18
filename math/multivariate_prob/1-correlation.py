#!/usr/bin/env python3
"""Calculate a correlation matrix from a covariance matrix."""
import numpy as np

def correlation(C):
    """Calculate the correlation matrix for a given covariance matrix.

    Args:
        C (np.ndarray): Covariance matrix of shape (d, d).
    Raises:
        TypeError: if C is not a np.ndarray.
        ValueError: if C is not a square (d, d) matrix.
    Returns:
        np.ndarray: Correlation matrix (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d = C.shape[0]
    stddev = np.sqrt(np.diag(C))
    denom = np.outer(stddev, stddev)
    corr = np.zeros((d, d))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = C / denom
        corr[np.isnan(corr)] = 0
    return corr
