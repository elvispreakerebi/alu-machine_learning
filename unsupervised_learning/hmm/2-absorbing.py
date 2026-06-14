#!/usr/bin/env python3
"""Determine whether a Markov chain is absorbing."""

import numpy as np


def absorbing(P):
    """
    Determine if a Markov chain is absorbing.

    Args:
        P: Standard transition matrix of shape (n, n).

    Returns:
        True if the chain is absorbing, False otherwise.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]
    diag = np.diag(P)

    if not np.any(diag == 1):
        return False
    if np.all(diag == 1):
        return True

    t = np.where(diag != 1)[0][0]
    R = P[t:, :t]
    Q = P[t:, t:]

    try:
        F = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    except Exception:
        return False

    if np.all(F @ R == 0):
        return False

    return True
