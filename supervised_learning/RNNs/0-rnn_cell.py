#!/usr/bin/env python3
"""
Simple RNN cell module.
"""

import numpy as np


class RNNCell:
    """
    One step of a vanilla RNN with tanh hidden state and softmax outputs.
    """

    def __init__(self, i, h, o):
        """
        Build weight matrices and biases for one RNN cell.

        Args:
            i (int): Input feature dimensionality.
            h (int): Hidden state dimensionality.
            o (int): Output dimensionality.

        Sets:
            Wh (np.ndarray): Shape (i + h, h). Applied as concat @ Wh + bh.
            Wy (np.ndarray): Shape (h, o). Applied as h_next @ Wy + by.
            bh (np.ndarray): Shape (1, h), biases for the hidden update.
            by (np.ndarray): Shape (1, o), biases for the output logits.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Run forward propagation for a single time step.

        Args:
            h_prev (np.ndarray): Previous hidden states of shape (m, h).
            x_t (np.ndarray): Inputs for this step of shape (m, i).

        Returns:
            tuple: (h_next, y) where h_next has shape (m, h) with tanh
                activation, and y has shape (m, o) with softmax rows.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        logits = h_next @ self.Wy + self.by
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return h_next, y
