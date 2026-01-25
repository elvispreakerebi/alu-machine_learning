#!/usr/bin/env python3
"""
Module for updating variables using gradient descent with momentum.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): A numpy array containing the variable to be updated.
        grad (numpy.ndarray): A numpy array containing the gradient of var.
        v (numpy.ndarray): The previous first moment of var.

    Returns:
        tuple: A tuple containing:
            - var (numpy.ndarray): The updated variable.
            - v (numpy.ndarray): The new moment.
    """
    v = beta1 * v + grad
    var = var - alpha * v
    return var, v
