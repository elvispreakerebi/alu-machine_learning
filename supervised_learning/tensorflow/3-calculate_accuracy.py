#!/usr/bin/env python3
"""
Module for calculating the accuracy of a prediction.
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (tf.Tensor): A placeholder for the labels of the input data.
        y_pred (tf.Tensor): A tensor containing the network's predictions.

    Returns:
        tf.Tensor: A tensor containing the decimal accuracy of the prediction.
    """
    # Get the predicted class (index with highest value)
    y_pred_class = tf.argmax(y_pred, axis=1)
    
    # Get the true class (index with value 1 in one-hot encoding)
    y_true_class = tf.argmax(y, axis=1)
    
    # Compare predicted and true classes
    correct_predictions = tf.equal(y_pred_class, y_true_class)
    
    # Convert boolean to float and calculate mean (accuracy)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy
