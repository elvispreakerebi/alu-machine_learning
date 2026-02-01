#!/usr/bin/env python3
"""
Module for creating confusion matrices.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and predictions.

    Args:
        labels (numpy.ndarray): One-hot array of shape (m, classes) containing
                               the correct labels for each data point.
        logits (numpy.ndarray): One-hot array of shape (m, classes) containing
                               the predicted labels.

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes) with row
                      indices representing the correct labels and column indices
                      representing the predicted labels.
    """
    m, classes = labels.shape
    
    # Convert one-hot encoded arrays to class indices
    # argmax returns the index of the maximum value (the class)
    true_labels = np.argmax(labels, axis=1)  # Shape: (m,)
    pred_labels = np.argmax(logits, axis=1)  # Shape: (m,)
    
    # Initialize confusion matrix
    confusion = np.zeros((classes, classes), dtype=np.float64)
    
    # Count occurrences: confusion[i, j] = count of true class i predicted as class j
    # Use np.add.at for efficient accumulation
    np.add.at(confusion, (true_labels, pred_labels), 1)
    
    return confusion
