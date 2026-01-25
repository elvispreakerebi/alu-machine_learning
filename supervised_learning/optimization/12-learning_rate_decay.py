#!/usr/bin/env python3
"""
Module for learning rate decay using inverse time decay in TensorFlow.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using inverse time decay
    in a stepwise fashion.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate at which
                           alpha will decay.
        global_step (tf.Variable): The number of passes of gradient descent that
                                   have elapsed.
        decay_step (int): The number of passes of gradient descent that
                         should occur before alpha is decayed further.

    Returns:
        tf.Tensor: The learning rate decay operation.
    """
    # Calculate the number of decay steps that have occurred (stepwise)
    # Convert decay_step to float for division
    decay_step_float = tf.cast(decay_step, tf.float32)
    decay_steps = tf.floor(tf.divide(tf.cast(global_step, tf.float32), decay_step_float))
    
    # Apply inverse time decay: alpha / (1 + decay_rate * decay_steps)
    # Convert alpha and decay_rate to tensors to ensure proper computation
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)
    decay_rate_tensor = tf.constant(decay_rate, dtype=tf.float32)
    
    # Compute the decayed learning rate
    updated_alpha = tf.divide(alpha_tensor, 
                              tf.add(1.0, 
                                    tf.multiply(decay_rate_tensor, decay_steps)))
    
    return updated_alpha
