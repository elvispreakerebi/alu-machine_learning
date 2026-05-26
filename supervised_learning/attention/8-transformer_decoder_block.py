#!/usr/bin/env python3
"""Single Transformer decoder block."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block: masked self-attention, cross-attention, FFN.

    Attributes:
        mha1: Masked multi-head self-attention.
        mha2: Multi-head cross-attention over encoder output.
        dense_hidden: First feed-forward dense layer (ReLU).
        dense_output: Second feed-forward dense layer (``dm`` units).
        layernorm1: Layer normalization after self-attention.
        layernorm2: Layer normalization after cross-attention.
        layernorm3: Layer normalization after feed-forward.
        dropout1: Dropout after self-attention.
        dropout2: Dropout after cross-attention.
        dropout3: Dropout after feed-forward.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize DecoderBlock.

        Args:
            dm: Model dimensionality.
            h: Number of attention heads.
            hidden: Hidden units in the feed-forward layer.
            drop_rate: Dropout rate (default 0.1).
        """
        super().__init__(name='decoder_block')
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the decoder block.

        Args:
            x: Decoder input ``(batch, target_seq_len, dm)``.
            encoder_output: Encoder output ``(batch, input_seq_len, dm)``.
            training: Whether the model is in training mode.
            look_ahead_mask: Mask for the first multi-head attention layer.
            padding_mask: Mask for the second multi-head attention layer.

        Returns:
            Tensor of shape ``(batch, target_seq_len, dm)``.
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn, training=training)
        return self.layernorm3(out2 + ffn)
