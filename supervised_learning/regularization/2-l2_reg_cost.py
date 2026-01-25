#!/usr/bin/env python3
"""
Module for calculating L2 regularization cost in TensorFlow.
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tf.Tensor): Tensor containing the cost of the network without
                         L2 regularization.

    Returns:
        tf.Tensor: Tensor containing the cost of the network accounting for
                  L2 regularization.
    """
    # Get all regularization losses from the graph
    # These are automatically added when using kernel_regularizer in layers
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # Sum all regularization losses (if any)
    if reg_losses:
        total_reg_loss = tf.add_n(reg_losses)
        # Add regularization losses to the original cost
        cost_with_l2 = cost + total_reg_loss
    else:
        # No regularization losses found, return original cost
        cost_with_l2 = cost
    
    return cost_with_l2
