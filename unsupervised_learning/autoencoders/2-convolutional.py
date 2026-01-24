#!/usr/bin/env python3
"""
Module for creating a convolutional autoencoder.
"""

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims (tuple): A tuple of integers containing the dimensions of
                           the model input.
        filters (list): A list containing the number of filters for each
                       convolutional layer in the encoder.
        latent_dims (tuple): A tuple of integers containing the dimensions of
                            the latent space representation.

    Returns:
        tuple: A tuple containing:
            - encoder (keras.Model): The encoder model.
            - decoder (keras.Model): The decoder model.
            - auto (keras.Model): The full autoencoder model.
    """
    # Encoder
    encoder_input = K.Input(shape=input_dims, name='encoder_input')
    x = encoder_input
    
    # Build encoder layers
    for i, num_filters in enumerate(filters):
        x = K.layers.Conv2D(
            num_filters,
            (3, 3),
            activation='relu',
            padding='same',
            name='encoder_conv_{}'.format(i)
        )(x)
        x = K.layers.MaxPooling2D((2, 2), padding='same', name='encoder_pool_{}'.format(i))(x)
    
    # Latent representation
    encoded = x
    
    # Create encoder model
    encoder = K.Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = K.Input(shape=latent_dims, name='decoder_input')
    x = decoder_input
    
    # Reverse filters for decoder
    reversed_filters = filters[::-1]
    
    # Build decoder layers
    num_decoder_layers = len(reversed_filters)
    for i, num_filters in enumerate(reversed_filters):
        if i < num_decoder_layers - 2:
            # Regular decoder layers: same padding, relu, upsampling
            x = K.layers.Conv2D(
                num_filters,
                (3, 3),
                activation='relu',
                padding='same',
                name='decoder_conv_{}'.format(i)
            )(x)
            x = K.layers.UpSampling2D((2, 2), name='decoder_upsample_{}'.format(i))(x)
        elif i == num_decoder_layers - 2:
            # Second to last: valid padding, relu, upsampling
            x = K.layers.Conv2D(
                num_filters,
                (3, 3),
                activation='relu',
                padding='valid',
                name='decoder_conv_{}'.format(i)
            )(x)
            x = K.layers.UpSampling2D((2, 2), name='decoder_upsample_{}'.format(i))(x)
    
    # Last layer: same filters as input channels, sigmoid, no upsampling
    x = K.layers.Conv2D(
        input_dims[2],
        (3, 3),
        activation='sigmoid',
        padding='same',
        name='decoder_output'
    )(x)
    
    decoded = x
    
    # Create decoder model
    decoder = K.Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder (encoder -> decoder)
    auto_input = encoder_input
    auto_output = decoder(encoder(auto_input))
    auto = K.Model(auto_input, auto_output, name='autoencoder')
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
