#!/usr/bin/env python3
"""
Neuron module for binary classification.
"""

import numpy as np


class Neuron:
    """
    A single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the weights vector.

        Returns:
            numpy.ndarray: The weights vector.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias.

        Returns:
            float: The bias value.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activated output.

        Returns:
            float: The activated output value.
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
