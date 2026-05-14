#!/usr/bin/env python3
"""
Forward half of a bidirectional RNN cell (weights for full bidirection).
"""

import numpy as np


class BidirectionalCell:
    """
    Holds forward/backward hidden weights and output weights; ``forward``
    updates only the forward-direction hidden state.
    """

    def __init__(self, i, h, o):
        """
        Initialize parameters.

        Args:
            i (int): Input size.
            h (int): Hidden size per direction.
            o (int): Output logits dimension (Wy maps concatenated 2h → o).

        Shapes:
            Whf, Whb: (i + h, h), multiplied on the right.
            Wy: (2 * h, o).
            bhf, bhb: (1, h); by: (1, o).
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        One forward-direction step (same recurrence as a vanilla RNN cell).

        Args:
            h_prev (np.ndarray): Previous forward hidden state (m, h).
            x_t (np.ndarray): Input (m, i).

        Returns:
            np.ndarray: Next forward hidden state (m, h), tanh activated.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(concat @ self.Whf + self.bhf)
