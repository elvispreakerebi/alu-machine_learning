#!/usr/bin/env python3
"""Intersection calculation for binomial data with priors."""
import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """Calculate the intersection of this data with prior probabilities.

    Args:
        x (int): Number of successes observed.
        n (int): Total number of trials.
        P (np.ndarray): 1D array, various hypothetical probabilities.
        Pr (np.ndarray): 1D array of same shape, prior beliefs for each P.
    Returns:
        np.ndarray: intersection values for all probabilities in P.
    Raises (in this order):
        ValueError/TypeError: As specified per assignment.
    """
    if not (isinstance(n, int)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not (isinstance(x, int)) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not (isinstance(P, np.ndarray) and P.ndim == 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if not (isinstance(Pr, np.ndarray) and Pr.shape == P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    like = likelihood(x, n, P)
    return like * Pr
