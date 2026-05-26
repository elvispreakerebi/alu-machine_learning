#!/usr/bin/env python3
"""Full Transformer decoder stack."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Transformer decoder: embeddings, positional encoding, and N blocks.

    Attributes:
        N: Number of decoder blocks.
        dm: Model dimensionality.
        embedding: Target token embedding layer.
        positional_encoding: Sinusoidal position encodings (numpy).
        blocks: List of ``DecoderBlock`` layers.
        dropout: Dropout applied after positional encoding.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize the Transformer decoder.

        Args:
            N: Number of decoder blocks.
            dm: Model dimensionality.
            h: Number of attention heads per block.
            hidden: Feed-forward hidden size in each block.
            target_vocab: Target vocabulary size.
            max_seq_len: Maximum sequence length for positional encodings.
            drop_rate: Dropout rate (default 0.1).
        """
        super().__init__(name='decoder')
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Decode token indices with positional signal and stacked blocks.

        Args:
            x: Target token indices ``(batch, target_seq_len)``.
            encoder_output: Encoder output ``(batch, input_seq_len, dm)``.
            training: Whether the model is in training mode.
            look_ahead_mask: Mask for masked self-attention.
            padding_mask: Mask for encoder-decoder attention.

        Returns:
            Tensor of shape ``(batch, target_seq_len, dm)``.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )

        return x
