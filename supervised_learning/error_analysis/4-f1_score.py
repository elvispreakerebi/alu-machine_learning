#!/usr/bin/env python3
"""
Module for calculating F1 score from confusion matrices.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                  where row indices represent the correct labels
                                  and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the F1 score of
                      each class.
    """
    # Calculate sensitivity (recall) and precision
    sens = sensitivity(confusion)  # Shape: (classes,)
    prec = precision(confusion)  # Shape: (classes,)
    
    # F1 score = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero (when precision + recall = 0)
    denominator = prec + sens
    f1_scores = np.divide(
        2 * prec * sens,
        denominator,
        out=np.zeros_like(prec, dtype=np.float64),
        where=(denominator != 0)
    )
    
    return f1_scores
