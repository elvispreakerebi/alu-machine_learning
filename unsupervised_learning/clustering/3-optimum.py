#!/usr/bin/env python3
"""Search cluster counts by intra-cluster variance."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test cluster sizes from kmin to kmax and track variance improvement.

    Args:
        X: Data set of shape (n, d).
        kmin: Minimum number of clusters to test (inclusive).
        kmax: Maximum number of clusters to test (inclusive); defaults to n.
        iterations: Maximum K-means iterations per k.

    Returns:
        Tuple (results, d_vars) where results is a list of (C, clss) from
        K-means for each k, and d_vars lists variance reduction vs kmin;
        or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n = X.shape[0]
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmax < kmin + 1:
        return None, None
    if kmax > n:
        return None, None

    results = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))

    base = variance(X, results[0][0])
    if base is None:
        return None, None

    d_vars = []
    for C, _ in results:
        v = variance(X, C)
        if v is None:
            return None, None
        d_vars.append(base - v)

    return results, d_vars
