#!/usr/bin/env python3
"""
Variational Autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.
    Args:
        input_dims (int): dimension of the input
        hidden_layers (list): number of nodes for each encoder hidden layer
        latent_dims (int): dimension of latent space
    Returns:
        encoder, decoder, auto
    """
    encoder_input = keras.Input(shape=(input_dims,))

    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mu, log_var = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(encoder_input, [z, mu, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)

    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.Model(decoder_input, decoder_output)

    z, mu, log_var = encoder(encoder_input)
    reconstructed = decoder(z)

    auto = keras.Model(encoder_input, reconstructed)

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_input, reconstructed)
    reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)

    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var - keras.backend.square(mu) - keras.backend.exp(log_var),
        axis=1)

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer="adam")

    return encoder, decoder, auto
