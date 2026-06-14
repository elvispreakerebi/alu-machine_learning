#!/usr/bin/env python3
"""Expectation-Maximization algorithm for a Gaussian Mixture Model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform expectation maximization for a GMM.

    Args:
        X: Data set of shape (n, d).
        k: Number of clusters.
        iterations: Maximum number of EM iterations.
        tol: Log likelihood tolerance for early stopping.
        verbose: Whether to print log likelihood every 10 iterations.

    Returns:
        Tuple (pi, m, S, g, l) of priors, means, covariances, posteriors,
        and log likelihood, or (None, None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    if X.shape[0] < k:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    g, log_l = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(log_l))

    prev_l = log_l
    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, log_l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, log_l))

        if abs(log_l - prev_l) <= tol:
            if verbose and i % 10 != 0:
                print("Log Likelihood after {} iterations: {:.5f}".format(
                    i, log_l))
            break

        prev_l = log_l

    return pi, m, S, g, log_l
