#!/usr/bin/env python3
"""Forward algorithm for a hidden Markov model."""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a hidden Markov model.

    Args:
        Observation: Observation indices of shape (T,).
        Emission: Emission probabilities of shape (N, M).
        Transition: Transition probabilities of shape (N, N).
        Initial: Initial state probabilities of shape (N, 1).

    Returns:
        Tuple (P, F) where P is the likelihood and F is shape (N, T),
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

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = Emission[:, Observation[t]] * (Transition.T @ F[:, t - 1])

    return np.sum(F[:, -1]), F
