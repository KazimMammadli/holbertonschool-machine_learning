#!/usr/bin/env python3
"""
Transfer learning on CIFAR-10 using EfficientNetB0.
Saves the trained and fine-tuned model as cifar10.h5.
"""

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for EfficientNetB0"""
    X_p = K.applications.efficientnet.preprocess_input(X)  # [-1,1] preprocessing
    Y_p = to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Input and resizing
    input_tensor = layers.Input(shape=(32, 32, 3))
    resize_layer = layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(input_tensor)

    # Load pre-trained EfficientNetB0 without top layers
    base_model = K.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=resize_layer,
        pooling='avg'
    )
    base_model.trainable = False  # freeze base for first phase

    # Add custom classification head
    x = base_model.output
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    # Compile model for first phase
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    print("--- Training the new classification head ---")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=5,
        validation_data=(x_test, y_test)
    )

    # Fine-tune last layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-40]:  # freeze first layers, unfreeze last 40
        layer.trainable = False

    # Compile for fine-tuning
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint_cb = K.callbacks.ModelCheckpoint(
        'cifar10.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
    early_stopping_cb = K.callbacks.EarlyStopping(
        patience=5,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    print("\n--- Fine-tuning the model ---")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    print("\n--- Saving final model to cifar10.h5 ---")
    model.save('cifar10.h5')

