#!/usr/bin/env python3
"""Steady state probabilities of a regular Markov chain."""

import numpy as np


def regular(P):
    """
    Determine the steady state probabilities of a regular Markov chain.

    Args:
        P: Transition matrix of shape (n, n).

    Returns:
        Steady state probabilities of shape (1, n), or None on failure.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    if not (P > 0).all():
        return None

    n = P.shape[0]
    Q = P - np.eye(n)
    M = np.vstack((Q.T[:-1], np.ones(n)))
    b = np.vstack((np.zeros((n - 1, 1)), np.array([[1]])))

    return np.linalg.solve(M, b).T
