#!/usr/bin/env python3
"""
Module for calculating specificity from confusion matrices.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity (True Negative Rate) for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                  where row indices represent the correct labels
                                  and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the specificity of
                      each class.
    """
    classes = confusion.shape[0]
    
    # Total number of instances
    total = np.sum(confusion)
    
    # True Positives: diagonal elements
    true_positives = np.diag(confusion)  # Shape: (classes,)
    
    # False Positives: sum of column i excluding the diagonal
    # Instances predicted as class i but are actually not class i
    column_sums = np.sum(confusion, axis=0)  # Total predicted as each class
    false_positives = column_sums - true_positives  # Shape: (classes,)
    
    # False Negatives: sum of row i excluding the diagonal
    # Instances that are class i but were predicted as not class i
    row_sums = np.sum(confusion, axis=1)  # Total instances of each true class
    false_negatives = row_sums - true_positives  # Shape: (classes,)
    
    # True Negatives: instances that are NOT class i and were correctly predicted as NOT class i
    # TN = Total - TP - FP - FN
    # Or: TN = sum of all confusion[j, k] where j != i and k != i
    true_negatives = total - true_positives - false_positives - false_negatives  # Shape: (classes,)
    
    # Specificity = TN / (TN + FP)
    # TN + FP = total instances that are NOT class i = total - row_sums[i]
    tn_plus_fp = total - row_sums  # Shape: (classes,)
    
    # Handle division by zero (if all instances are class i)
    specificity_scores = np.divide(
        true_negatives,
        tn_plus_fp,
        out=np.zeros_like(true_negatives, dtype=np.float64),
        where=(tn_plus_fp != 0)
    )
    
    return specificity_scores
