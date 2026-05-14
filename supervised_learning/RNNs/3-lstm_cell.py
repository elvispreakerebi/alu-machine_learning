#!/usr/bin/env python3
"""
Long Short-Term Memory (LSTM) cell module.
"""

import numpy as np


class LSTMCell:
    """
    One LSTM step: forget / input / candidate / output gates, cell update,
    hidden projection, and softmax readout.
    """

    def __init__(self, i, h, o):
        """
        Initialize LSTM parameters.

        Args:
            i (int): Input size.
            h (int): Hidden and cell state size.
            o (int): Softmax output dimension.

        Weight matrices have shape (i + h, h) and multiply on the right:
        concat @ W + b. Wy has shape (h, o). Biases have shapes (1, h) or
        (1, o) for by.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward pass for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden state (m, h).
            c_prev (np.ndarray): Previous cell state (m, h).
            x_t (np.ndarray): Input (m, i).

        Returns:
            tuple: (h_next, c_next, y) with y row-softmax of shape (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        forget = self._sigmoid(concat @ self.Wf + self.bf)
        update = self._sigmoid(concat @ self.Wu + self.bu)
        cand = np.tanh(concat @ self.Wc + self.bc)
        out_gate = self._sigmoid(concat @ self.Wo + self.bo)

        c_next = forget * c_prev + update * cand
        h_next = out_gate * np.tanh(c_next)

        logits = h_next @ self.Wy + self.by
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        y = ez / np.sum(ez, axis=1, keepdims=True)

        return h_next, c_next, y

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
