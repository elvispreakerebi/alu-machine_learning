#!/usr/bin/env python3
"""Additive (Bahdanau-style) attention for seq2seq decoding."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Additive alignment: Dense W/U plus Dense V on tanh(score)."""

    def __init__(self, units, **kwargs):
        super().__init__(name="self_attention", **kwargs)
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Apply additive attention scores over encoder time steps.

        Args:
            s_prev: ``(batch, units)`` prior decoder hidden state.
            hidden_states: ``(batch, input_seq_len, units)`` encoder outputs.

        Returns:
            Tuple ``(context, weights)``: ``context`` has shape ``(batch,
            units)``, ``weights`` ``(batch, input_seq_len, 1)``.
        """
        projected_state = tf.expand_dims(self.W(s_prev), axis=1)
        projected_enc = self.U(hidden_states)
        energies = self.V(tf.tanh(projected_state + projected_enc))
        weights = tf.nn.softmax(energies, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
