#!/usr/bin/env python3
"""
Module for calculating sensitivity (recall) from confusion matrices.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                  where row indices represent the correct labels
                                  and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the sensitivity of
                      each class.
    """
    classes = confusion.shape[0]
    
    # True Positives: diagonal elements (correctly predicted instances)
    # confusion[i, i] = number of times class i was correctly predicted
    true_positives = np.diag(confusion)  # Shape: (classes,)
    
    # Total instances of each true class: sum of each row
    # This is TP + FN (true positives + false negatives)
    row_sums = np.sum(confusion, axis=1)  # Shape: (classes,)
    
    # Sensitivity (Recall) = TP / (TP + FN) = TP / row_sum
    # Handle division by zero (if a class has no instances)
    sensitivity_scores = np.divide(
        true_positives,
        row_sums,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=(row_sums != 0)
    )
    
    return sensitivity_scores
