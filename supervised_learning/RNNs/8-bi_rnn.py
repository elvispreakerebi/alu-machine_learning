#!/usr/bin/env python3
"""
Bidirectional RNN unrolled forward pass.
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Run a bidirectional RNN over the input sequence.

    The forward direction scans t=0..T-1 starting from ``h_0``; the backward
    direction scans t=T-1..0 starting from ``h_t``. At each step ``s`` the
    stored hidden vector is ``concat(forward[s], backward[s])`` (shape ``2h``).

    Args:
        bi_cell: BidirectionalCell with ``forward``, ``backward``, and
            ``output`` methods.
        X (np.ndarray): Inputs of shape (t, m, i).
        h_0 (np.ndarray): Initial forward hidden state (m, h).
        h_t (np.ndarray): Initial backward hidden state at sequence end (m, h).

    Returns:
        tuple:
            H (np.ndarray): Shape (t, m, 2 * h), concatenated hiddens per step.
            Y (np.ndarray): Shape (t, m, o), softmax outputs from ``output``.
    """
    t = X.shape[0]
    h_dim = h_0.shape[1]

    forward_h = np.zeros((t, X.shape[1], h_dim))
    hf = h_0
    for s in range(t):
        hf = bi_cell.forward(hf, X[s])
        forward_h[s] = hf

    backward_h = np.zeros((t, X.shape[1], h_dim))
    hb = h_t
    for s in range(t - 1, -1, -1):
        h_prev = bi_cell.backward(hb, X[s])
        backward_h[s] = h_prev
        hb = h_prev

    H = np.concatenate((forward_h, backward_h), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
