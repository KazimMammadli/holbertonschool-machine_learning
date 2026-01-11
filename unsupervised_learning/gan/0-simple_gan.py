#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
    ):
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Generator loss and optimizer
        self.generator.loss = (
            lambda x:
            tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
        )

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # Discriminator loss and optimizer
        self.discriminator.loss = (
            lambda x, y:
            tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
            + tf.keras.losses.MeanSquaredError()(
                y, -1 * tf.ones(y.shape)
            )
        )

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss,
        )

    # Generator of fake samples
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size

        return self.generator(
            self.latent_generator(size),
            training=training,
        )

    # Generator of real samples
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size

        sorted_indices = tf.range(
            tf.shape(self.real_examples)[0]
        )

        random_indices = tf.random.shuffle(
            sorted_indices
        )[:size]

        return tf.gather(
            self.real_examples,
            random_indices,
        )

    # Overloading train_step()
    def train_step(self, useless_argument):

        # 1) Train discriminator
        for _ in range(self.disc_iter):
            x_real = self.get_real_sample(
                self.batch_size
            )

            x_fake = self.get_fake_sample(
                self.batch_size,
                training=True,
            )

            with tf.GradientTape() as tape_d:
                d_real = self.discriminator(
                    x_real,
                    training=True,
                )

                d_fake = self.discriminator(
                    x_fake,
                    training=True,
                )

                discr_loss = self.discriminator.loss(
                    d_real,
                    d_fake,
                )

            grads_d = tape_d.gradient(
                discr_loss,
                self.discriminator.trainable_variables,
            )

            self.discriminator.optimizer.apply_gradients(
                zip(
                    grads_d,
                    self.discriminator.trainable_variables,
                )
            )

        # 2) Train generator
        with tf.GradientTape() as tape_g:
            x_fake = self.get_fake_sample(
                self.batch_size,
                training=True,
            )

            d_fake = self.discriminator(
                x_fake,
                training=False,
            )

            gen_loss = self.generator.loss(
                d_fake
            )

        grads_g = tape_g.gradient(
            gen_loss,
            self.generator.trainable_variables,
        )

        self.generator.optimizer.apply_gradients(
            zip(
                grads_g,
                self.generator.trainable_variables,
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
        }
