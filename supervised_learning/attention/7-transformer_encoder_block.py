#!/usr/bin/env python3
"""Single Transformer encoder block (attention + feed-forward)."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Transformer encoder block: multi-head self-attention and feed-forward net.

    Attributes:
        mha: Multi-head attention layer.
        dense_hidden: First feed-forward dense layer (ReLU).
        dense_output: Second feed-forward dense layer (``dm`` units).
        layernorm1: Layer normalization after attention.
        layernorm2: Layer normalization after feed-forward.
        dropout1: Dropout after attention.
        dropout2: Dropout after feed-forward.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize EncoderBlock.

        Args:
            dm: Model dimensionality.
            h: Number of attention heads.
            hidden: Hidden units in the feed-forward layer.
            drop_rate: Dropout rate (default 0.1).
        """
        super().__init__(name='encoder_block')
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass through the encoder block.

        Args:
            x: Input tensor of shape ``(batch, input_seq_len, dm)``.
            training: Whether the model is in training mode.
            mask: Optional mask for multi-head attention.

        Returns:
            Tensor of shape ``(batch, input_seq_len, dm)``.
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn = self.dense_hidden(out1)
        ffn = self.dense_output(ffn)
        ffn = self.dropout2(ffn, training=training)
        return self.layernorm2(out1 + ffn)
