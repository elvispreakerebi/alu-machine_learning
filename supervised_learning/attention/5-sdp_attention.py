#!/usr/bin/env python3
"""Scaled dot-product attention (Transformer)."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor ``(..., seq_len_q, dk)``.
        K: Key tensor ``(..., seq_len_v, dk)``.
        V: Value tensor ``(..., seq_len_v, dv)``.
        mask: Optional mask broadcastable to ``(..., seq_len_q, seq_len_v)``.

    Returns:
        Tuple ``(output, weights)`` where ``output`` has shape
        ``(..., seq_len_q, dv)`` and ``weights`` has shape
        ``(..., seq_len_q, seq_len_v)``.
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)

    if mask is not None:
        scaled += mask * -1e9

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
