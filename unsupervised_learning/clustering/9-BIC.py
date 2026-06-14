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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmax < kmin:
        return None, None, None, None
    if kmax > n:
        return None, None, None, None

    size = kmax - kmin + 1
    log_l = np.zeros(size)
    bics = np.zeros(size)
    best_k = None
    best_result = None
    best_bic = np.inf

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, _, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        log_l[idx] = ll
        p = (k - 1) + k * d + k * (d * (d + 1) // 2)
        bic = p * np.log(n) - 2 * ll
        bics[idx] = bic

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_l, bics
