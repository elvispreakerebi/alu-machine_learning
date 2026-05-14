#!/usr/bin/env python3
"""
Bidirectional RNN cell: hidden updates and readout from concatenated states.
"""

import numpy as np


class BidirectionalCell:
    """
    Holds forward/backward hidden weights and output weights for a Bi-RNN.
    """

    def __init__(self, i, h, o):
        """
        Initialize parameters.

        Args:
            i (int): Input size.
            h (int): Hidden size per direction.
            o (int): Output size for Wy (concatenated 2h → o).

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
        Forward-direction step: next hidden from previous hidden.

        Args:
            h_prev (np.ndarray): Previous forward hidden state (m, h).
            x_t (np.ndarray): Input (m, i).

        Returns:
            np.ndarray: Next forward hidden state (m, h).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(concat @ self.Whf + self.bhf)

    def backward(self, h_next, x_t):
        """
        Backward-direction step: previous hidden from next hidden.

        Args:
            h_next (np.ndarray): Next hidden state in the backward scan (m, h).
            x_t (np.ndarray): Input (m, i).

        Returns:
            np.ndarray: Previous hidden state (m, h).
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(concat @ self.Whb + self.bhb)

    def output(self, H):
        """
        Apply the output layer with softmax at every time step.

        Args:
            H (np.ndarray): Concatenated forward/backward hiddens (t, m, 2*h).

        Returns:
            np.ndarray: Probabilities (t, m, o).
        """
        logits = H @ self.Wy + self.by
        z = logits - np.max(logits, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
