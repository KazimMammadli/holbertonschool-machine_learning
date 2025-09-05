#!/usr/bin/env python3
"""Early stoppping and patience parameters."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Return the History object generated after training the model."""
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          early_stopping=early_stopping, patience=patience,
                          shuffle=shuffle, verbose=verbose)
    return history
