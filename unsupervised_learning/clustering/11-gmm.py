#!/usr/bin/env python3
"""Gaussian Mixture Model using sklearn."""

import sklearn.mixture


def gmm(X, k):
    """
    Calculate a GMM from a dataset.

    Args:
        X: Dataset of shape (n, d).
        k: Number of clusters.

    Returns:
        Tuple (pi, m, S, clss, bic) of priors, means, covariances,
        cluster assignments, and BIC value.
    """
    gm = sklearn.mixture.GaussianMixture(n_components=k)
    clss = gm.fit_predict(X)

    return gm.weights_, gm.means_, gm.covariances_, clss, gm.bic(X)
