#!/usr/bin/env python3
"""GRU encoder with Embeddings for attention / MT."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encoder embedding word indices through an Embedding followed by GRU:

    Args:
        vocab: Number of vocabulary items (embedding input dimension).
        embedding: Size of embedding vectors (`Embedding(..., embedding)`).
        units: Hidden size of GRU.
        batch: Batch size used to build the initial zeros state tensor.
    """

    def __init__(self, vocab, embedding, units, batch, **kwargs):
        super().__init__(name="rnn_encoder", **kwargs)
        self.batch = batch
        self.units = units
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

    def initialize_hidden_state(self):
        """Return ``(batch, units)`` float32 zeros."""
        return tf.zeros((self.batch, self.units), dtype=tf.float32)

    def call(self, x, initial):
        """
        Args:
            x: ``(batch, input_seq_len)`` int vocabulary indices.
            initial: ``(batch, units)`` initial GRU state.

        Returns:
            Tuple ``(outputs, hidden)``: GRU outputs for the whole sequence and
            the final hidden state.
        """
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded, initial_state=initial)
        return outputs, hidden
