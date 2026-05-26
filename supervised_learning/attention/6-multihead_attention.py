#!/usr/bin/env python3
"""Multi-head attention layer for Transformer models."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention using parallel scaled dot-product attention heads.

    Attributes:
        h: Number of attention heads.
        dm: Model dimensionality.
        depth: Dimension per head (``dm // h``).
        Wq: Query projection layer.
        Wk: Key projection layer.
        Wv: Value projection layer.
        linear: Output projection layer.
    """

    def __init__(self, dm, h):
        """
        Initialize MultiHeadAttention.

        Args:
            dm: Model dimensionality (must be divisible by ``h``).
            h: Number of parallel attention heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Apply multi-head attention to the inputs.

        Args:
            Q: Tensor of shape ``(batch, seq_len_q, dk)``.
            K: Tensor of shape ``(batch, seq_len_v, dk)``.
            V: Tensor of shape ``(batch, seq_len_v, dv)``.
            mask: Optional attention mask (often ``None``).

        Returns:
            Tuple ``(output, weights)`` where ``output`` has shape
            ``(batch, seq_len_q, dm)`` and ``weights`` has shape
            ``(batch, h, seq_len_q, seq_len_v)``.
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self._split_heads(Q, batch_size)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)

        attention, weights = sdp_attention(Q, K, V, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)
        return output, weights

    def _split_heads(self, x, batch_size):
        """Reshape to ``(batch, h, seq_len, depth)``."""
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
