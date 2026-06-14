#!/usr/bin/env python3
"""K-means cluster centroid initialization."""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means.

    Each centroid dimension is drawn independently from a uniform
    distribution on [min(X[:, j]), max(X[:, j])] for dimension j.

    Args:
        X: Dataset of shape (n, d).
        k: Number of clusters (positive integer).

    Returns:
        Centroids of shape (k, d), or None on failure.
    """
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if X.shape[0] < k:
        return None

    return np.random.uniform(
        X.min(axis=0), X.max(axis=0), size=(k, X.shape[1])
    )
