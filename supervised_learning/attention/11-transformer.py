#!/usr/bin/env python3
"""Transformer sequence-to-sequence model."""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Full Transformer: encoder, decoder, and vocabulary projection.

    Attributes:
        encoder: Encoder stack.
        decoder: Decoder stack.
        linear: Final dense layer to target vocabulary logits.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initialize the Transformer model.

        Args:
            N: Number of blocks in encoder and decoder.
            dm: Model dimensionality.
            h: Number of attention heads.
            hidden: Feed-forward hidden units per block.
            input_vocab: Source vocabulary size.
            target_vocab: Target vocabulary size.
            max_seq_input: Maximum source sequence length.
            max_seq_target: Maximum target sequence length.
            drop_rate: Dropout rate (default 0.1).
        """
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Forward pass through the Transformer.

        Args:
            inputs: Source token indices ``(batch, input_seq_len)``.
            target: Target token indices ``(batch, target_seq_len)``.
            training: Whether the model is in training mode.
            encoder_mask: Padding mask for the encoder.
            look_ahead_mask: Look-ahead mask for the decoder.
            decoder_mask: Padding mask for encoder-decoder attention.

        Returns:
            Logits of shape ``(batch, target_seq_len, target_vocab)``.
        """
        enc_output = self.encoder(
            inputs, training=training, mask=encoder_mask)
        dec_output = self.decoder(
            target,
            enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask,
        )
        return self.linear(dec_output)
