#!/usr/bin/env python3
"""
One-hot decoding module.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): A one-hot encoded numpy array with shape
                                (classes, m), where classes is the maximum
                                number of classes and m is the number of
                                examples.

    Returns:
        numpy.ndarray: A numpy array with shape (m,) containing the numeric
                      labels for each example, or None on failure.
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2:
        return None
    if one_hot.shape[0] == 0 or one_hot.shape[1] == 0:
        return None

    # Check if each column sums to 1 (valid one-hot encoding)
    column_sums = np.sum(one_hot, axis=0)
    if not np.allclose(column_sums, 1):
        return None

    # Find the index of the maximum value in each column (the class label)
    labels = np.argmax(one_hot, axis=0)

    return labels
