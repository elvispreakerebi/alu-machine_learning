#!/usr/bin/env python3
"""Full Transformer encoder stack."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Transformer encoder: embeddings, positional encoding, and N blocks.

    Attributes:
        N: Number of encoder blocks.
        dm: Model dimensionality.
        embedding: Input token embedding layer.
        positional_encoding: Sinusoidal position encodings (numpy).
        blocks: List of ``EncoderBlock`` layers.
        dropout: Dropout applied after positional encoding.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize the Transformer encoder.

        Args:
            N: Number of encoder blocks.
            dm: Model dimensionality.
            h: Number of attention heads per block.
            hidden: Feed-forward hidden size in each block.
            input_vocab: Input vocabulary size.
            max_seq_len: Maximum sequence length for positional encodings.
            drop_rate: Dropout rate (default 0.1).
        """
        super().__init__(name='encoder')
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Encode token indices with positional signal and stacked blocks.

        Args:
            x: Token indices ``(batch, input_seq_len)``.
            training: Whether the model is in training mode.
            mask: Optional attention mask for encoder blocks.

        Returns:
            Tensor of shape ``(batch, input_seq_len, dm)``.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
