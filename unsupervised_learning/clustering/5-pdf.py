#!/usr/bin/env python3
"""Multivariate Gaussian probability density function."""

import numpy as np


def pdf(X, m, S):
    """
    Calculate the PDF of a Gaussian distribution at each data point.

    Args:
        X: Data points of shape (n, d).
        m: Mean of the distribution of shape (d,).
        S: Covariance matrix of shape (d, d).

    Returns:
        PDF values of shape (n,), or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d:
        return None

    diff = X - m
    S_inv = np.linalg.inv(S)
    mahal = np.sum((diff @ S_inv) * diff, axis=1)
    det = np.linalg.det(S)
    norm = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    P = norm * np.exp(-0.5 * mahal)

    return np.maximum(P, 1e-300)
