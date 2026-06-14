#!/usr/bin/env python3
"""Total intra-cluster variance for K-means."""

import numpy as np


def variance(X, C):
    """
    Calculate total intra-cluster variance for a dataset.

    Each point is assigned to its nearest centroid in C; variance is the
    sum of squared Euclidean distances to those centroids.

    Args:
        X: Data set of shape (n, d).
        C: Centroid means of shape (k, d).

    Returns:
        Total variance as a float, or None on failure.
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[0] == 0 or C.shape[0] == 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    dists = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
    clss = np.argmin(dists, axis=1)
    return np.sum(dists[np.arange(X.shape[0]), clss])
