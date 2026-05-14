#!/usr/bin/env python3
"""
Gated Recurrent Unit (GRU) cell module.
"""

import numpy as np


class GRUCell:
    """
    One GRU step with sigmoid gates, tanh candidate state, and softmax output.
    """

    def __init__(self, i, h, o):
        """
        Initialize GRU parameters.

        Args:
            i (int): Input size.
            h (int): Hidden state size.
            o (int): Output size (softmax dimension).

        Weights multiply on the right: gate_x @ W + b.
        Shapes: Wz, Wr, Wh are (i + h, h); Wy is (h, o);
        bz, br, bh are (1, h); by is (1, o).
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward pass for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden states (m, h).
            x_t (np.ndarray): Inputs (m, i).

        Returns:
            tuple: (h_next, y) with h_next (m, h) and softmax y (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(concat @ self.Wz + self.bz)
        r = self._sigmoid(concat @ self.Wr + self.br)

        cand_concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(cand_concat @ self.Wh + self.bh)

        h_next = (1.0 - z) * h_prev + z * h_tilde

        logits = h_next @ self.Wy + self.by
        z_soft = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z_soft)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-x))
