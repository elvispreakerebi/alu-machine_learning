#!/usr/bin/env python3
"""K-means clustering using sklearn."""

import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: Dataset of shape (n, d).
        k: Number of clusters.

    Returns:
        Tuple (C, clss) of centroids and cluster assignments.
    """
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    clss = kmeans_model.fit_predict(X)

    return kmeans_model.cluster_centers_, clss
