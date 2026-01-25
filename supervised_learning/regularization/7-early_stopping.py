#!/usr/bin/env python3
"""
Module for early stopping in neural network training.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early based on validation cost.

    Args:
        cost (float): The current validation cost of the neural network.
        opt_cost (float): The lowest recorded validation cost of the neural network.
        threshold (float): The threshold used for early stopping.
        patience (int): The patience count used for early stopping.
        count (int): The count of how long the threshold has not been met.

    Returns:
        tuple: A tuple containing:
            - bool: Whether the network should be stopped early.
            - int: The updated count.
    """
    # Calculate the improvement: how much better is the current cost vs optimal?
    improvement = opt_cost - cost
    
    # Check if improvement exceeds threshold
    if improvement > threshold:
        # Significant improvement: reset count
        updated_count = 0
    else:
        # Insufficient improvement: increment count
        updated_count = count + 1
    
    # Check if we should stop early
    # Stop if count has reached or exceeded patience
    should_stop = updated_count >= patience
    
    return should_stop, updated_count
