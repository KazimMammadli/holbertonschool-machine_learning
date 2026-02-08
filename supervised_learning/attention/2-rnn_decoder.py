#!/usr/bin/env python3
"""RNN Decoder Module"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder class for machine translation that uses attention
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Decoder

        Args:
            vocab: integer representing the size of the output vocabulary
            embedding: integer representing the dimensionality of the
                      embedding vector
            units: integer representing the number of hidden units in
                  the RNN cell
            batch: integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass through the decoder

        Args:
            x: tensor of shape (batch, 1) containing the previous word
               in the target sequence as an index of the target
               vocabulary
            s_prev: tensor of shape (batch, units) containing the
                   previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                          containing the outputs of the encoder

        Returns:
            y: tensor of shape (batch, vocab) containing the output
               word as a one hot vector in the target vocabulary
            s: tensor of shape (batch, units) containing the new
               decoder hidden state
        """
        # Initialize self-attention
        attention = SelfAttention(s_prev.shape[1])

        # Calculate context vector using attention
        context, _ = attention(s_prev, hidden_states)

        # Embed the input word
        x = self.embedding(x)

        # Concatenate context vector with embedded input
        # context shape: (batch, units), x shape: (batch, 1, embedding)
        # Expand context to (batch, 1, units) then concat
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        # Pass through GRU
        output, s = self.gru(x, initial_state=s_prev)

        # Remove the time dimension
        output = tf.squeeze(output, axis=1)

        # Apply final dense layer
        y = self.F(output)

        return y, s
