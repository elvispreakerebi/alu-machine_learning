#!/usr/bin/env python3
"""Viterbi algorithm for a hidden Markov model."""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculate the most likely sequence of hidden states for an HMM.

    Args:
        Observation: Observation indices of shape (T,).
        Emission: Emission probabilities of shape (N, M).
        Transition: Transition probabilities of shape (N, N).
        Initial: Initial state probabilities of shape (N, 1).

    Returns:
        Tuple (path, P) where path is a list of length T and P is the
        probability of that path, or (None, None) on failure.
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

    mu = np.zeros((N, T))
    trail = np.zeros((N, T))
    path = []

    mu[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for n in range(N):
            probs = (
                Emission[n, Observation[t]] * Transition[:, n] * mu[:, t - 1]
            )
            trail[n, t] = np.argmax(probs)
            mu[n, t] = np.max(probs)

    P = np.max(mu[:, -1])
    path.append(int(np.argmax(mu[:, -1])))

    for t in range(T - 1, 0, -1):
        path.insert(0, int(trail[path[0], t]))

    return path, P
