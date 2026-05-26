#!/usr/bin/env python3
"""Attention-based GRU decoder step for neural MT."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decoder layer using additive attention over encoder states.

    Attributes:
        embedding: Target vocabulary embedding layer.
        gru: Decoder recurrent layer.
        F: Output projection to vocabulary size.
        attention: Alignment model over encoder hidden states.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNNDecoder.

        Args:
            vocab: Size of the output vocabulary.
            embedding: Dimensionality of the embedding vector.
            units: Number of hidden units in the GRU cell.
            batch: Batch size (unused; kept for API compatibility).
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Perform one decoder step with attention.

        Args:
            x: Tensor of shape (batch, 1) with previous target word indices.
            s_prev: Tensor of shape (batch, units) with previous decoder state.
            hidden_states: Tensor of shape (batch, input_seq_len, units)
                with encoder outputs.

        Returns:
            Tuple (y, s):
                y: Tensor of shape (batch, vocab) with output logits.
                s: Tensor of shape (batch, units) with new decoder state.
        """
        context, _ = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, 1)
        x = self.embedding(x)
        x = tf.concat([context, x], axis=-1)
        _, s = self.gru(x, initial_state=s_prev)
        y = self.F(s)
        return y, s
