#!/usr/bin/env python3
"""Bayesian Information Criterion for Gaussian Mixture Models."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC.

    Args:
        X: Data set of shape (n, d).
        kmin: Minimum number of clusters to test (inclusive).
        kmax: Maximum number of clusters to test (inclusive); defaults to n.
        iterations: Maximum EM iterations per cluster count.
        tol: EM log likelihood tolerance.
        verbose: Whether EM should print progress information.

    Returns:
        Tuple (best_k, best_result, l, b) where best_result is (pi, m, S),
        l and b are arrays of log likelihoods and BIC values per k tested,
        or (None, None, None, None) on failure.
    """
    fail = (None, None, None, None)
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(kmin) is not int or
            kmin < 1):
        return fail

    n, d = X.shape
    if kmax is None:
        kmax = n
    if (
            type(kmax) is not int or
            kmax < 1 or
            kmax < kmin + 1 or
            type(iterations) is not int or
            iterations < 1 or
            type(tol) is not float or
            tol < 0 or
            type(verbose) is not bool):
        return fail

    log_l = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return fail

        results.append((pi, m, S))
        log_l.append(ll)
        p = k * (d + 2) * (d + 1) / 2 - 1
        bics.append(np.log(n) * p - 2 * ll)

    best_index = np.argmin(bics)
    return (
        kmin + best_index,
        results[best_index],
        np.array(log_l),
        np.array(bics),
    )
