#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims (int): input dimension
        hidden_layers (list): hidden layer sizes for encoder
        latent_dims (int): latent space dimension

    Returns:
        encoder, decoder, auto
    """
    K = keras.backend

    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    z_mean = keras.layers.Dense(
        latent_dims, activation=None, name="z_mean"
    )(x)

    z_log_var = keras.layers.Dense(
        latent_dims, activation=None, name="z_log_var"
    )(x)

    def sampling(args):
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    encoder = keras.models.Model(
        encoder_input, [z, z_mean, z_log_var], name="encoder"
    )

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)

    decoder_output = keras.layers.Dense(
        input_dims, activation="sigmoid"
    )(x)

    decoder = keras.models.Model(
        decoder_input, decoder_output, name="decoder"
    )

    z, mu, log_var = encoder(encoder_input)
    reconstructed = decoder(z)

    auto = keras.models.Model(
        encoder_input, reconstructed, name="vae"
    )

    reconstruction_loss = K.sum(
        keras.losses.binary_crossentropy(encoder_input, reconstructed),
        axis=1
    )

    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mu) - K.exp(log_var),
        axis=1
    )

    auto.add_loss(K.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer="adam")

    return encoder, decoder, auto
