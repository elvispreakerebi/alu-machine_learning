#!/usr/bin/env python3
"""
Module for updating variables using the RMSProp optimization algorithm.
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): A numpy array containing the variable to be updated.
        grad (numpy.ndarray): A numpy array containing the gradient of var.
        s (numpy.ndarray): The previous second moment of var.

    Returns:
        tuple: A tuple containing:
            - var (numpy.ndarray): The updated variable.
            - s (numpy.ndarray): The new second moment.
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
