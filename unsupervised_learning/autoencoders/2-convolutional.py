#!/usr/bin/env python3
"""Convolutional Autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates an autoencoder model.
    Args:
        input_dims (tuple): Tuple of integers containing the dimensions of
        the model input
        filters (list) : Containing the number of filters for each convolutional
        layer in the encoder
        latent_dims (tuple): Tuple of integers containing the dimensions of the
        latent space representation
    Returns:
        encoder (Model): Encoder model
        decoder (Model): Decoder model
        auto (Model): Full autoencoder model
    """
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(f, kernel_size=(3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for i, f in enumerate(filters[::-1], 1):
        x = keras.layers.Conv2D(f, kernel_size=(3, 3),
                                padding='valid'
                                if i == len(filters) else 'same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    
    decoder_output = keras.layers.Conv2D(input_dims[-1], kernel_size=(3, 3),
                                         padding='same',
                                         activation='sigmoid')(x)

    decoder = keras.Model(decoder_input, decoder_output)

    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
