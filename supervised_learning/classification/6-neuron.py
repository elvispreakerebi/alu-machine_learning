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

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m), where m is the
                              number of examples.
            A (numpy.ndarray): Activated output with shape (1, m), containing
                              the activated output of the neuron for each
                              example.

        Returns:
            float: The cost of the model.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.
            Y (numpy.ndarray): Correct labels with shape (1, m), containing the
                              correct labels for the input data.

        Returns:
            tuple: A tuple containing:
                - prediction (numpy.ndarray): Predicted labels with shape (1, m),
                  where label values are 1 if output >= 0.5, 0 otherwise.
                - cost (float): The cost of the network.
        """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.
            Y (numpy.ndarray): Correct labels with shape (1, m), containing the
                              correct labels for the input data.
            A (numpy.ndarray): Activated output with shape (1, m), containing
                              the activated output of the neuron for each
                              example.
            alpha (float): The learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        dW = (1 / m) * np.dot(A - Y, X.T)
        db = (1 / m) * np.sum(A - Y)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.
            Y (numpy.ndarray): Correct labels with shape (1, m), containing the
                              correct labels for the input data.
            iterations (int): The number of iterations to train over. Defaults
                             to 5000.
            alpha (float): The learning rate. Defaults to 0.05.

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float.
            ValueError: If iterations is not positive or alpha is not positive.

        Returns:
            tuple: A tuple containing:
                - prediction (numpy.ndarray): Predicted labels with shape (1, m),
                  where label values are 1 if output >= 0.5, 0 otherwise.
                - cost (float): The cost of the network.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
