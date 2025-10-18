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

    def pdf(self, x):
        """Calculate the PDF value at data point x.

        Args:
            x (np.ndarray): shape (d, 1) data point.
        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x.shape is not (d, 1).
        Returns:
            float: The value of the Multivariate Normal PDF at x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        diff = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        det = np.linalg.det(self.cov)
        norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
        exp_arg = -0.5 * np.dot(diff.T, np.dot(cov_inv, diff))
        exp_val = np.exp(exp_arg)
        return float(norm * exp_val)
