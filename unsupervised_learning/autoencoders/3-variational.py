#!/usr/bin/env python3
"""
Module for creating a variational autoencoder.
"""

import tensorflow.keras as K
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each
                             hidden layer in the encoder.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        tuple: A tuple containing:
            - encoder (keras.Model): The encoder model.
            - decoder (keras.Model): The decoder model.
            - auto (keras.Model): The full autoencoder model.
    """
    # Encoder
    encoder_input = K.Input(shape=(input_dims,), name='encoder_input')
    x = encoder_input
    
    # Build encoder hidden layers
    for i, nodes in enumerate(hidden_layers):
        x = K.layers.Dense(nodes, activation='relu', name='encoder_layer_{}'.format(i))(x)
    
    # Mean layer (no activation)
    mu = K.layers.Dense(latent_dims, activation=None, name='mu')(x)
    
    # Log variance layer (no activation)
    log_sig = K.layers.Dense(latent_dims, activation=None, name='log_sig')(x)
    
    # Sampling layer
    def sampling(args):
        mu, log_sig = args
        epsilon = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(log_sig / 2) * epsilon
    
    encoded = K.layers.Lambda(sampling, output_shape=(latent_dims,), name='sampling')([mu, log_sig])
    
    # Create encoder model (outputs: latent, mu, log_sig)
    encoder = K.Model(encoder_input, [encoded, mu, log_sig], name='encoder')
    
    # Decoder
    decoder_input = K.Input(shape=(latent_dims,), name='decoder_input')
    x = decoder_input
    
    # Build decoder layers (reverse of encoder hidden layers)
    reversed_layers = hidden_layers[::-1]
    for i, nodes in enumerate(reversed_layers):
        x = K.layers.Dense(nodes, activation='relu', name='decoder_layer_{}'.format(i))(x)
    
    # Output layer with sigmoid activation
    decoded = K.layers.Dense(input_dims, activation='sigmoid', name='decoder_output')(x)
    
    # Create decoder model
    decoder = K.Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder (encoder -> decoder)
    auto_input = encoder_input
    encoded_output, mu_output, log_sig_output = encoder(auto_input)
    auto_output = decoder(encoded_output)
    auto = K.Model(auto_input, auto_output, name='autoencoder')
    
    # Add KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(1 + log_sig_output - tf.square(mu_output) - tf.exp(log_sig_output))
    auto.add_loss(kl_loss)
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
