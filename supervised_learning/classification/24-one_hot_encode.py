#!/usr/bin/env python3
"""
One-hot encoding module.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): A numpy array with shape (m,) containing numeric
                          class labels, where m is the number of examples.
        classes (int): The maximum number of classes found in Y.

    Returns:
        numpy.ndarray: A one-hot encoding of Y with shape (classes, m), or
                      None on failure.
    """
    if not isinstance(Y, np.ndarray):
        return None
    if len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 1:
        return None
    if np.any(Y < 0) or np.any(Y >= classes):
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1

    return one_hot
