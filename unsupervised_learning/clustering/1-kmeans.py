#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: Dataset of shape (n, d).
        k: Number of clusters.
        iterations: Maximum number of iterations.

    Returns:
        Tuple (C, clss) of centroids and cluster assignments, or (None, None).
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if X.shape[0] < k:
        return None, None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, (k, d))

    for _ in range(iterations):
        dists = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
        clss = np.argmin(dists, axis=1)

        one_hot = np.zeros((n, k))
        one_hot[np.arange(n), clss] = 1
        counts = one_hot.sum(axis=0)
        C_new = np.zeros((k, d))
        nonempty = counts > 0
        C_new[nonempty] = (one_hot.T @ X)[nonempty] / counts[nonempty, np.newaxis]

        empty = np.where(counts == 0)[0]
        if empty.size > 0:
            C_new[empty] = np.random.uniform(low, high, (empty.size, d))

        if np.allclose(C_new, C):
            return C_new, clss

        C = C_new

    dists = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
    clss = np.argmin(dists, axis=1)
    return C, clss
