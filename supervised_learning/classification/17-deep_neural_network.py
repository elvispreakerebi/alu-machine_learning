#!/usr/bin/env python3
"""
Deep Neural Network module for binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    A deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize a DeepNeuralNetwork instance.

        Args:
            nx (int): The number of input features.
            layers (list): A list representing the number of nodes in each layer
                          of the network.

        Raises:
            TypeError: If nx is not an integer or layers is not a list or
                      contains non-positive integers.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases for each layer
        for l in range(1, self.__L + 1):
            if l == 1:
                # First layer: input size is nx
                n_prev = nx
            else:
                # Subsequent layers: input size is previous layer size
                n_prev = layers[l - 2]

            # He et al. initialization: W ~ N(0, sqrt(2/n_prev))
            self.__weights['W{}'.format(l)] = np.random.normal(
                0, np.sqrt(2 / n_prev), (layers[l - 1], n_prev))
            # Biases initialized to zeros
            self.__weights['b{}'.format(l)] = np.zeros((layers[l - 1], 1))

    @property
    def L(self):
        """
        Getter for the number of layers.

        Returns:
            int: The number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter for the cache dictionary.

        Returns:
            dict: A dictionary to hold all intermediary values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter for the weights dictionary.

        Returns:
            dict: A dictionary to hold all weights and biases of the network.
        """
        return self.__weights
