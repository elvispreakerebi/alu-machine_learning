#!/usr/bin/env python3
"""
Module for updating variables using the Adam optimization algorithm.
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight used for the first moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): A numpy array containing the variable to be updated.
        grad (numpy.ndarray): A numpy array containing the gradient of var.
        v (numpy.ndarray): The previous first moment of var.
        s (numpy.ndarray): The previous second moment of var.
        t (int): The time step used for bias correction.

    Returns:
        tuple: A tuple containing:
            - var (numpy.ndarray): The updated variable.
            - v (numpy.ndarray): The new first moment.
            - s (numpy.ndarray): The new second moment.
    """
    # Update the first moment (momentum)
    v = beta1 * v + (1 - beta1) * grad
    
    # Update the second moment (RMSProp)
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    
    # Apply bias correction to the first moment
    v_corrected = v / (1 - beta1 ** t)
    
    # Apply bias correction to the second moment
    s_corrected = s / (1 - beta2 ** t)
    
    # Update the variable
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    
    return var, v, s
