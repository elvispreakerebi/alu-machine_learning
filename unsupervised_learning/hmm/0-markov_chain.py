#!/usr/bin/env python3
"""Markov chain state probability after t iterations."""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determine the probability of each state after t Markov chain iterations.

    Args:
        P: Transition matrix of shape (n, n).
        s: Initial state probabilities of shape (1, n).
        t: Number of iterations.

    Returns:
        State probabilities of shape (1, n), or None on failure.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]

    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if s.shape != (1, n):
        return None

    if type(t) is not int or t < 0:
        return None

    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    if not np.isclose(np.sum(s), 1):
        return None

    return np.matmul(s, np.linalg.matrix_power(P, t))
