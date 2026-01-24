#!/usr/bin/env python3
"""
Module for creating a sparse autoencoder.
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder model.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each
                             hidden layer in the encoder.
        latent_dims (int): The dimensions of the latent space representation.
        lambtha (float): The regularization parameter used for L1 regularization
                        on the encoded output.

    Returns:
        tuple: A tuple containing:
            - encoder (keras.Model): The encoder model.
            - decoder (keras.Model): The decoder model.
            - auto (keras.Model): The sparse autoencoder model.
    """
    # Encoder
    encoder_input = K.Input(shape=(input_dims,), name='encoder_input')
    x = encoder_input
    
    # Build encoder layers
    for i, nodes in enumerate(hidden_layers):
        x = K.layers.Dense(nodes, activation='relu', name='encoder_layer_{}'.format(i))(x)
    
    # Latent representation with L1 regularization
    encoded = K.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=K.regularizers.l1(lambtha),
        name='latent'
    )(x)
    
    # Create encoder model
    encoder = K.Model(encoder_input, encoded, name='encoder')
    
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
    auto_output = decoder(encoder(auto_input))
    auto = K.Model(auto_input, auto_output, name='autoencoder')
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
