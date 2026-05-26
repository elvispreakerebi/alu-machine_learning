#!/usr/bin/env python3
"""Attention-based GRU decoder step for neural MT."""

import tensorflow as tf

SelfAttention = __import__("1-self_attention").SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Attention context + token embedding, GRU, then vocabulary logits."""

    def __init__(self, vocab, embedding, units, batch, **kwargs):
        """
        Args:
            vocab: Target vocabulary size (embedding inputs + logits dim).
            embedding: Embedding dimension for target tokens.
            units: Hidden size (must match encoder / attention).
            batch: Batch size (mirrors ``RNNEncoder`` constructor).
        """
        super().__init__(name="rnn_decoder", **kwargs)
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding,
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(vocab)
        self._attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Args:
            x: ``(batch, 1)`` target-token indices.
            s_prev: ``(batch, units)`` previous decoder state.
            hidden_states: ``(batch, input_seq_len, units)`` encoder outputs.

        Returns:
            ``y`` logits ``(batch, vocab)`` and new state ``s`` shaped
            ``(batch, units)``.
        """
        context, _ = self._attention(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        embedded = self.embedding(x)
        decoder_input = tf.concat([context, embedded], axis=-1)
        _, s = self.gru(decoder_input, initial_state=s_prev)
        y = self.F(s)
        return y, s
