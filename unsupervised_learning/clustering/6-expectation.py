#!/usr/bin/env python3
"""Expectation step for a Gaussian Mixture Model."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X: Data set of shape (n, d).
        pi: Priors for each cluster of shape (k,).
        m: Centroid means for each cluster of shape (k, d).
        S: Covariance matrices for each cluster of shape (k, d, d).

    Returns:
        Tuple (g, l) where g contains posterior probabilities of shape (k, n)
        and l is the total log likelihood, or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    if np.any(pi < 0):
        return None, None
    if not np.all(np.isfinite(pi)):
        return None, None
    if not np.allclose(np.sum(pi), 1):
        return None, None

    prob = np.zeros((k, n))
    for j in range(k):
        P = pdf(X, m[j], S[j])
        if P is None:
            return None, None
        prob[j] = pi[j] * P

    total = prob.sum(axis=0)
    g = prob / total[np.newaxis, :]

    return g, np.sum(np.log(total))
