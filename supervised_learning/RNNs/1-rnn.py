#!/usr/bin/env python3
"""
Unrolled forward pass for a simple RNN.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Run forward propagation through `t` time steps using one RNN cell.

    Args:
        rnn_cell: Instance of RNNCell.
        X (np.ndarray): Inputs of shape (t, m, i).
        h_0 (np.ndarray): Initial hidden state of shape (m, h).

    Returns:
        tuple:
            H (np.ndarray): All hidden states, shape (t + 1, m, h). Row 0 is
                h_0; row s + 1 is the state after step s.
            Y (np.ndarray): Outputs at each step, shape (t, m, o).
    """
    t = X.shape[0]
    m, h_dim = h_0.shape
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h_dim))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for s in range(t):
        h_next, y = rnn_cell.forward(H[s], X[s])
        H[s + 1] = h_next
        Y[s] = y

    return H, Y
