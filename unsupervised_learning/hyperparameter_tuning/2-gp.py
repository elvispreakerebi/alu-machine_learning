#!/usr/bin/env python3
"""
Module for Gaussian Process implementation.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """
    
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize a Gaussian Process.

        Args:
            X_init (numpy.ndarray): Array of shape (t, 1) representing the inputs
                                   already sampled with the black-box function.
            Y_init (numpy.ndarray): Array of shape (t, 1) representing the outputs
                                   of the black-box function for each input in X_init.
            l (float): The length parameter for the kernel. Defaults to 1.
            sigma_f (float): The standard deviation given to the output of the
                            black-box function. Defaults to 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        
        # Compute the covariance kernel matrix for the initial inputs
        self.K = self.kernel(X_init, X_init)
    
    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices using
        the Radial Basis Function (RBF).

        Args:
            X1 (numpy.ndarray): Array of shape (m, 1).
            X2 (numpy.ndarray): Array of shape (n, 1).

        Returns:
            numpy.ndarray: The covariance kernel matrix of shape (m, n).
        """
        # Compute pairwise squared distances
        # X1: (m, 1), X2: (n, 1)
        # We want to compute (x1_i - x2_j)^2 for all i, j
        # Using broadcasting: (m, 1) - (1, n) = (m, n)
        sq_dist = np.square(X1 - X2.T)
        
        # Apply RBF kernel formula: sigma_f^2 * exp(-0.5 * sq_dist / l^2)
        K = self.sigma_f ** 2 * np.exp(-0.5 * sq_dist / (self.l ** 2))
        
        return K
    
    def predict(self, X_s):
        """
        Predict the mean and standard deviation of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Array of shape (s, 1) containing all of the points
                                whose mean and standard deviation should be calculated.

        Returns:
            tuple: A tuple containing:
                - mu (numpy.ndarray): Array of shape (s,) containing the mean for
                  each point in X_s.
                - sigma (numpy.ndarray): Array of shape (s,) containing the variance
                  for each point in X_s.
        """
        # Compute covariance matrices
        # K_s: covariance between training points (self.X) and test points (X_s)
        # Shape: (t, s)
        K_s = self.kernel(self.X, X_s)
        
        # K_ss: covariance of test points with themselves
        # Shape: (s, s)
        K_ss = self.kernel(X_s, X_s)
        
        # Compute inverse of training covariance matrix
        # Shape: (t, t)
        K_inv = np.linalg.inv(self.K)
        
        # Compute mean: mu = K_s.T @ K_inv @ Y
        # K_s.T: (s, t), K_inv: (t, t), self.Y: (t, 1)
        # Result: (s, 1), then flatten to (s,)
        mu = (K_s.T @ K_inv @ self.Y).flatten()
        
        # Compute variance: sigma = diag(K_ss - K_s.T @ K_inv @ K_s)
        # K_s.T: (s, t), K_inv: (t, t), K_s: (t, s)
        # K_s.T @ K_inv @ K_s: (s, s)
        # K_ss - K_s.T @ K_inv @ K_s: (s, s)
        # diag(...): (s,)
        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)
        
        return mu, sigma
    
    def update(self, X_new, Y_new):
        """
        Update a Gaussian Process with new sample points.

        Args:
            X_new (numpy.ndarray): Array of shape (1,) representing the new sample point.
            Y_new (numpy.ndarray): Array of shape (1,) representing the new sample
                                  function value.
        """
        # Reshape X_new and Y_new to (1, 1) to match the format of X and Y
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)
        
        # Append new points to X and Y
        self.X = np.vstack([self.X, X_new])
        self.Y = np.vstack([self.Y, Y_new])
        
        # Recompute the covariance kernel matrix for all points
        self.K = self.kernel(self.X, self.X)
