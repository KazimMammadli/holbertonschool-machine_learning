#!/usr/bin/env python3
"""Vanilla AutoEncoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model.

    Args:
        input_dims (int): Dimension of the input
        hidden_layers (list): List of hidden layer sizes for encoder
        latent_dims (int): Dimension of the latent space

    Returns:
        encoder (Model): Encoder model
        decoder (Model): Decoder model
        auto (Model): Full autoencoder model
    """

    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(encoder_input, latent)


    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(decoder_input, decoder_output)

  
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
