#!/usr/bin/env python3
"""
Deep (stacked) RNN forward pass.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Run forward propagation through ``l`` stacked RNN layers over ``t`` steps.

    At each time step, layer 0 receives ``X[s]``; deeper layers receive the
    updated hidden state of the layer below. Each layer reads its previous
    hidden state from ``H[s, ell]``.

    Args:
        rnn_cells (list): Length ``l`` of RNNCell instances (aligned inputs).
        X (np.ndarray): Inputs of shape (t, m, i).
        h_0 (np.ndarray): Initial hiddens per layer, shape (l, m, h).

    Returns:
        tuple:
            H (np.ndarray): Shape (t + 1, l, m, h). ``H[0]`` equals ``h_0``;
                ``H[s + 1, ell]`` is layer ``ell`` after step ``s``.
            Y (np.ndarray): Shape (t, m, o), softmax outputs from the last cell.
    """
    l_len = len(rnn_cells)
    t = X.shape[0]
    m = X.shape[1]
    h_dim = h_0.shape[2]

    H = np.zeros((t + 1, l_len, m, h_dim))
    H[0] = h_0

    out_dim = rnn_cells[-1].Wy.shape[1]
    Y = np.zeros((t, m, out_dim))

    for s in range(t):
        layer_input = X[s]
        for ell in range(l_len):
            h_next, y_out = rnn_cells[ell].forward(H[s, ell], layer_input)
            H[s + 1, ell] = h_next
            layer_input = h_next
        Y[s] = y_out

    return H, Y
