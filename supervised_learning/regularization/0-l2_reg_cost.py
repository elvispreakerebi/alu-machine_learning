#!/usr/bin/env python3
"""
Module for calculating L2 regularization cost.
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: The cost of the network without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (dict): Dictionary of the weights and biases (numpy.ndarrays)
                       of the neural network.
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.

    Returns:
        float: The cost of the network accounting for L2 regularization.
    """
    # Calculate the sum of squares of all weights
    sum_squared_weights = 0
    
    # Iterate through all weight matrices (W1, W2, ..., WL)
    for i in range(1, L + 1):
        weight_key = 'W' + str(i)
        if weight_key in weights:
            # Sum of squares of all elements in the weight matrix
            sum_squared_weights += np.sum(np.square(weights[weight_key]))
    
    # Calculate L2 regularization term: (lambtha / (2 * m)) * sum of squares
    l2_reg_term = (lambtha / (2 * m)) * sum_squared_weights
    
    # Add regularization term to the original cost
    cost_with_l2 = cost + l2_reg_term
    
    return cost_with_l2
