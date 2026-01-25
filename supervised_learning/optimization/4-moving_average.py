#!/usr/bin/env python3
"""
Module for calculating weighted moving average with bias correction.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Args:
        data (list): A list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of data with bias correction.
    """
    moving_avg = []
    v = 0
    
    for i, value in enumerate(data, 1):
        # Calculate exponential moving average
        v = beta * v + (1 - beta) * value
        
        # Apply bias correction
        v_corrected = v / (1 - beta ** i)
        moving_avg.append(v_corrected)
    
    return moving_avg
