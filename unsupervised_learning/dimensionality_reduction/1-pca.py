#!/usr/bin/env python3
"""Principal Component Analysis with fixed dimensionality."""

import numpy as np


def pca(X, ndim):
    """
    Perform PCA on a dataset and return the transformed data.

    Args:
        X: Data set of shape (n, d).
        ndim: New dimensionality of the transformed X.

    Returns:
        Transformed data T of shape (n, ndim).
    """
    normX = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(normX, full_matrices=False)
    W = vh[:ndim].T
    return normX @ W
