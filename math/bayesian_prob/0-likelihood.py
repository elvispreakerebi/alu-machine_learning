#!/usr/bin/env python3
"""Likelihood calculation for binomial data with multiple probabilities."""
import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihoods of seeing x successes in n trials.

    Args:
        x (int): number of successes.
        n (int): total trials, must be positive.
        P (np.ndarray): 1D array, probabilities in [0, 1].
    Raises:
        ValueError: n, x checks; x in [0, n].
        TypeError: if P is not 1D np.ndarray.
        ValueError: if P contains values outside [0,1].
    Returns:
        np.ndarray: likelihoods for each probability in P.
    """
    if not (isinstance(n, int)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not (isinstance(x, int)) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not (isinstance(P, np.ndarray) and P.ndim == 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    fact_n = 1
    for i in range(1, n + 1):
        fact_n *= i
    fact_x = 1
    for i in range(1, x + 1):
        fact_x *= i
    fact_nx = 1
    for i in range(1, n - x + 1):
        fact_nx *= i
    comb = fact_n / (fact_x * fact_nx)
    return comb * (P ** x) * ((1 - P) ** (n - x))
