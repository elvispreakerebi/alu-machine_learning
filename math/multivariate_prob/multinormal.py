#!/usr/bin/env python3
"""Multivariate Normal distribution class."""
import numpy as np


class MultiNormal:
    """Represents a multivariate normal distribution."""

    def __init__(self, data):
        """Initialize a Multivariate Normal distribution from data.

        Args:
            data (np.ndarray): of shape (d, n), d = dimensions, n = samples.
        Raises:
            TypeError: If data is not a 2D np.ndarray.
            ValueError: If n < 2.
        Sets:
            mean: shape (d, 1)
            cov: shape (d, d)
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = np.dot(X_centered, X_centered.T) / (n - 1)
