#!/usr/bin/env python3
"""
11-transformer.py
Defines the Transformer model using the Encoder and Decoder classes.
"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer model that consists of an encoder, decoder, and final linear layer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """
        Initializes the Transformer model.

        Args:
            N (int): number of encoder/decoder blocks
            dm (int): model dimensionality
            h (int): number of attention heads
            hidden (int): number of hidden units in FC layer
            input_vocab (int): size of input vocabulary
            target_vocab (int): size of target vocabulary
            max_seq_input (int): maximum input sequence length
            max_seq_target (int): maximum target sequence length
            drop_rate (float): dropout rate
        """
        super().__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Forward pass for the transformer.

        Args:
            inputs (tensor): shape (batch, input_seq_len)
            target (tensor): shape (batch, target_seq_len)
            training (bool): training mode
            encoder_mask (tensor): encoder padding mask
            look_ahead_mask (tensor): look ahead mask for decoder
            decoder_mask (tensor): decoder padding mask

        Returns:
            tensor: shape (batch, target_seq_len, target_vocab)
        """
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output, _ = self.decoder(target, enc_output, training,
                                     look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
