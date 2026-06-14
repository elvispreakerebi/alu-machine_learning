#!/usr/bin/env python3
"""Principal Component Analysis."""

import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on a dataset and return the projection weights matrix.

    Args:
        X: Data set of shape (n, d) with mean 0 across data points.
        var: Fraction of variance to maintain.

    Returns:
        Weights matrix W of shape (d, nd).
    """
    _, s, vh = np.linalg.svd(X)
    cum = np.cumsum(s) / np.sum(s)
    r = np.argwhere(cum >= var)[0, 0]
    return vh[:r + 1].T
