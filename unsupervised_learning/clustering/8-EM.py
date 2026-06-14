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
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(k) is not int or
            k < 1 or
            type(iterations) is not int or
            iterations < 1 or
            type(tol) is not float or
            tol < 0 or
            type(verbose) is not bool):
        return (None, None, None, None, None)
    if X.shape[0] < k:
        return (None, None, None, None, None)

    pi, m, S = initialize(X, k)
    if pi is None:
        return (None, None, None, None, None)

    g, log_l = expectation(X, pi, m, S)
    if g is None:
        return (None, None, None, None, None)

    for iteration in range(iterations):
        prev_l = log_l
        if verbose and iteration % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(
                iteration, log_l))

        pi, m, S = maximization(X, g)
        if pi is None:
            return (None, None, None, None, None)

        g, log_l = expectation(X, pi, m, S)
        if g is None:
            return (None, None, None, None, None)

        if abs(log_l - prev_l) <= tol:
            break

    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}".format(
            iteration + 1, log_l))

    return pi, m, S, g, log_l
