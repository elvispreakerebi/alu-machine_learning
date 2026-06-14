#!/usr/bin/env python3
"""Baum-Welch algorithm for a hidden Markov model."""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a hidden Markov model.

    Args:
        Observation: Observation indices of shape (T,).
        Emission: Emission probabilities of shape (M, N).
        Transition: Transition probabilities of shape (M, M).
        Initial: Initial state probabilities of shape (M, 1).

    Returns:
        Tuple (P, F) where P is the likelihood and F is shape (M, T).
    """
    N = Emission.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = Emission[:, Observation[t]] * (
            Transition.T @ F[:, t - 1]
        )

    return np.sum(F[:, -1]), F


def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a hidden Markov model.

    Args:
        Observation: Observation indices of shape (T,).
        Emission: Emission probabilities of shape (M, N).
        Transition: Transition probabilities of shape (M, M).
        Initial: Initial state probabilities of shape (M, 1).

    Returns:
        Tuple (P, B) where P is the likelihood and B is shape (M, T).
    """
    N = Emission.shape[0]
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition
            * Emission[:, Observation[t + 1]]
            * B[:, t + 1],
            axis=1
        )

    P = np.sum(
        Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0]
    )
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Perform the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations: Observation indices of shape (T,).
        Transition: Initial transition probabilities of shape (M, M).
        Emission: Initial emission probabilities of shape (M, N).
        Initial: Initial state probabilities of shape (M, 1).
        iterations: Number of expectation-maximization iterations.

    Returns:
        Tuple (Transition, Emission) of converged parameters, or
        (None, None) on failure.
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    M, M2 = Transition.shape
    M3, N = Emission.shape
    T = Observations.shape[0]

    if M != M2 or M != M3 or Initial.shape != (M, 1):
        return None, None
    if np.any(Observations < 0) or np.any(Observations >= N):
        return None, None
    if not np.allclose(np.sum(Transition, axis=1), 1):
        return None, None
    if not np.allclose(np.sum(Emission, axis=1), 1):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    for _ in range(iterations):
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            numerator = (
                F[:, t][:, np.newaxis]
                * Transition
                * Emission[:, Observations[t + 1]][np.newaxis, :]
                * B[:, t + 1][np.newaxis, :]
            )
            xi[:, :, t] = numerator / np.sum(numerator)

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, axis=2) / np.sum(
            gamma, axis=1
        )[:, np.newaxis]

        gamma_full = np.hstack(
            (
                gamma,
                np.sum(xi[:, :, T - 2], axis=0)[:, np.newaxis],
            )
        )

        for j in range(N):
            Emission[:, j] = np.sum(
                gamma_full[:, Observations == j], axis=1
            )

        Emission = Emission / np.sum(gamma_full, axis=1)[:, np.newaxis]

    return Transition, Emission
