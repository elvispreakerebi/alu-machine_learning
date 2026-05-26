#!/usr/bin/env python3
"""Sinusoidal positional encodings for Transformer models."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Compute sinusoidal positional encoding vectors.

    For position ``pos`` and dimension ``i``:

    - even ``i``: ``sin(pos / 10000^(2i/dm))``
    - odd ``i``: ``cos(pos / 10000^(2i/dm))``

    Args:
        max_seq_len: Maximum sequence length.
        dm: Model depth (embedding dimension).

    Returns:
        ndarray of shape ``(max_seq_len, dm)`` with positional encodings.
    """
    pe = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
