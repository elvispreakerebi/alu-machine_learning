#!/usr/bin/env python3
"""
Module for calculating precision from confusion matrices.
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                  where row indices represent the correct labels
                                  and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the precision of
                      each class.
    """
    classes = confusion.shape[0]
    
    # True Positives: diagonal elements (correctly predicted instances)
    # confusion[i, i] = number of times class i was correctly predicted
    true_positives = np.diag(confusion)  # Shape: (classes,)
    
    # Total instances predicted as each class: sum of each column
    # This is TP + FP (true positives + false positives)
    column_sums = np.sum(confusion, axis=0)  # Shape: (classes,)
    
    # Precision = TP / (TP + FP) = TP / column_sum
    # Handle division by zero (if a class was never predicted)
    precision_scores = np.divide(
        true_positives,
        column_sums,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=(column_sums != 0)
    )
    
    return precision_scores
