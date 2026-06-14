#!/usr/bin/env python3
"""Backward algorithm for a hidden Markov model."""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a hidden Markov model.

    Args:
        Observation: Observation indices of shape (T,).
        Emission: Emission probabilities of shape (N, M).
        Transition: Transition probabilities of shape (N, N).
        Initial: Initial state probabilities of shape (N, 1).

    Returns:
        Tuple (P, B) where P is the likelihood and B is shape (N, T),
        or (None, None) on failure.
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None

    T = Observation.shape[0]

    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape

    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None

    if type(Initial) is not np.ndarray or Initial.shape != (N, 1):
        return None, None

    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = Transition @ (
            Emission[:, Observation[t + 1]] * B[:, t + 1]
        )

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
