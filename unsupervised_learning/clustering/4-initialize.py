#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize priors, means, and covariances for a Gaussian Mixture Model.

    Args:
        X: Data set of shape (n, d).
        k: Number of clusters.

    Returns:
        Tuple (pi, m, S) of priors, means, and covariance matrices, or
        (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    if X.shape[0] < k:
        return None, None, None

    d = X.shape[1]
    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
