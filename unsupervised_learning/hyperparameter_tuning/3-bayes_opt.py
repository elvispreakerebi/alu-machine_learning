#!/usr/bin/env python3
"""
Module for Bayesian Optimization implementation.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """
    
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization.

        Args:
            f: The black-box function to be optimized.
            X_init (numpy.ndarray): Array of shape (t, 1) representing the inputs
                                   already sampled with the black-box function.
            Y_init (numpy.ndarray): Array of shape (t, 1) representing the outputs
                                   of the black-box function for each input in X_init.
            bounds (tuple): Tuple of (min, max) representing the bounds of the space
                           in which to look for the optimal point.
            ac_samples (int): The number of samples that should be analyzed during
                             acquisition.
            l (float): The length parameter for the kernel. Defaults to 1.
            sigma_f (float): The standard deviation given to the output of the
                            black-box function. Defaults to 1.
            xsi (float): The exploration-exploitation factor for acquisition.
                        Defaults to 0.01.
            minimize (bool): Whether optimization should be performed for minimization
                            (True) or maximization (False). Defaults to True.
        """
        self.f = f
        self.xsi = xsi
        self.minimize = minimize
        
        # Initialize Gaussian Process
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        
        # Create acquisition sample points evenly spaced between bounds
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)
    
    def acquisition(self):
        """
        Calculate the next best sample location using the Expected Improvement
        acquisition function.

        Returns:
            tuple: A tuple containing:
                - X_next (numpy.ndarray): Array of shape (1,) representing the next
                  best sample point.
                - EI (numpy.ndarray): Array of shape (ac_samples,) containing the
                  expected improvement of each potential sample.
        """
        # Get predictions from Gaussian Process
        mu, sigma = self.gp.predict(self.X_s)
        
        # Get the best observed value
        if self.minimize:
            # For minimization: best is the minimum value
            f_best = np.min(self.gp.Y)
        else:
            # For maximization: best is the maximum value
            f_best = np.max(self.gp.Y)
        
        # Calculate Z for Expected Improvement
        # For minimization: Z = (f_best - μ - ξ) / σ
        # For maximization: Z = (μ - f_best - ξ) / σ
        if self.minimize:
            Z = (f_best - mu - self.xsi) / (sigma + 1e-9)  # Add small epsilon to avoid division by zero
        else:
            Z = (mu - f_best - self.xsi) / (sigma + 1e-9)
        
        # Calculate Expected Improvement: EI = σ * [Z * Φ(Z) + φ(Z)]
        # where Φ is the CDF and φ is the PDF of the standard normal distribution
        EI = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
        
        # Ensure EI is non-negative (should be, but handle numerical issues)
        EI = np.maximum(EI, 0)
        
        # Find the point with maximum Expected Improvement
        max_idx = np.argmax(EI)
        X_next = self.X_s[max_idx].flatten()
        
        return X_next, EI
