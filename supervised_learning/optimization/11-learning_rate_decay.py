#!/usr/bin/env python3
"""
Module for learning rate decay using inverse time decay.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate at which
                           alpha will decay.
        global_step (int): The number of passes of gradient descent that
                          have elapsed.
        decay_step (int): The number of passes of gradient descent that
                         should occur before alpha is decayed further.

    Returns:
        float: The updated value for alpha.
    """
    # Calculate the number of decay steps that have occurred
    decay_steps = np.floor(global_step / decay_step)
    
    # Apply inverse time decay: alpha / (1 + decay_rate * decay_steps)
    updated_alpha = alpha / (1 + decay_rate * decay_steps)
    
    return updated_alpha
