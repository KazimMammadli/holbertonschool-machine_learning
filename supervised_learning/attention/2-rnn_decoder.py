#!/usr/bin/env python3
"""
RNN Decoder module for machine translation with attention
"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder class using GRU and self-attention
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab (int): size of the output vocabulary
            embedding (int): dimensionality of the embedding vectors
            units (int): number of hidden units in the GRU
            batch (int): batch size
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(units=vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass of the decoder

        Args:
            x (tf.Tensor): previous target word
                           shape (batch, 1)
            s_prev (tf.Tensor): previous decoder hidden state
                                shape (batch, units)
            hidden_states (tf.Tensor): encoder outputs
                                       shape (batch, input_seq_len, units)

        Returns:
            y (tf.Tensor): output word scores
                           shape (batch, vocab)
            s (tf.Tensor): new decoder hidden state
                           shape (batch, units)
        """
        # Compute attention context
        context, _ = self.attention(s_prev, hidden_states)

        # Embed input word
        x = self.embedding(x)

        # Concatenate context and embedding
        context = tf.expand_dims(context, axis=1)
        x = tf.concat([context, x], axis=-1)

        # Pass through GRU
        outputs, s = self.gru(x, initial_state=s_prev)

        # Reshape output for Dense layer
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        # Final output
        y = self.F(outputs)

        return y, s
