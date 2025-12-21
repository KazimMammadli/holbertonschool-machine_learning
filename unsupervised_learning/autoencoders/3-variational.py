#!/usr/bin/env python3
"""Variational Autoencoder"""
from tensorflow import keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: Integer containing the dimensions of the model input
        hidden_layers: List containing the number of nodes for each
        hidden layer
        latent_dims: Integer containing the dimensions of the latent space
    Returns:
        encoder: The encoder model
        decoder: The decoder model
        auto: The full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Hidden layers for encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Latent space parameters
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    # Encoder model
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Hidden layers for decoder (reversed)
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Output layer
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Decoder model
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_outputs, name='autoencoder')

    def vae_loss(y_true, y_pred):
        # Reconstruction loss
        recon_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        recon_loss *= input_dims

        # KL divergence loss
        kl_loss = (1 + z_log_var - keras.backend.square(z_mean)
                   - keras.backend.exp(z_log_var))
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(recon_loss + kl_loss)

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
