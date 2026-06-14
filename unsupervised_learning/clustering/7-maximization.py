#!/usr/bin/env python3
"""Maximization step for a Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: Data set of shape (n, d).
        g: Posterior probabilities of shape (k, n).

    Returns:
        Tuple (pi, m, S) of updated priors, means, and covariance matrices,
        or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None

    N = g.sum(axis=1)
    if np.any(N == 0):
        return None, None, None

    pi = N / n
    m = (g @ X) / N[:, np.newaxis]

    S = np.zeros((k, d, d))
    for j in range(k):
        diff = X - m[j]
        S[j] = (g[j][:, np.newaxis] * diff).T @ diff / N[j]

    return pi, m, S
