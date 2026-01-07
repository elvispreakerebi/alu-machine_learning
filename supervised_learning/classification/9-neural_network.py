#!/usr/bin/env python3
"""
Neural Network module for binary classification with one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """
    A neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork instance.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer or nodes is not an integer.
            ValueError: If nx is less than 1 or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for the weights vector of the hidden layer.

        Returns:
            numpy.ndarray: The weights vector.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for the bias of the hidden layer.

        Returns:
            numpy.ndarray: The bias vector.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for the activated output of the hidden layer.

        Returns:
            float: The activated output value.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for the weights vector of the output neuron.

        Returns:
            numpy.ndarray: The weights vector.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for the bias of the output neuron.

        Returns:
            float: The bias value.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for the activated output of the output neuron.

        Returns:
            float: The activated output value.
        """
        return self.__A2
