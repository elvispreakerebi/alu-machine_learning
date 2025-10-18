#!/usr/bin/env python3
"""Marginal probability calculation for binomial data and priors."""
import numpy as np
intersection = __import__('1-intersection').intersection

def marginal(x, n, P, Pr):
    """Calculate the marginal probability of obtaining this data.

    Args:
        x (int): number of successes observed
        n (int): number of trials
        P (np.ndarray): 1D array of hypothetical probabilities
        Pr (np.ndarray): 1D array of priors, same shape as P
    Returns:
        float: marginal probability for x, n, all P/Pr
    Raises (same order/messages as intersection):
        ValueError/TypeError as appropriate
    """
    # Error checks, must match order in intersection
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
    inter = intersection(x, n, P, Pr)
    return np.sum(inter)
