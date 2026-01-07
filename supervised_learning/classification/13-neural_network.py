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

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.

        Returns:
            tuple: A tuple containing:
                - A1 (numpy.ndarray): The activated output of the hidden layer
                  with shape (nodes, m).
                - A2 (numpy.ndarray): The activated output of the output neuron
                  with shape (1, m).
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

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
        Evaluate the neural network's predictions.

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
        A1, A2 = self.forward_prop(X)
        prediction = (A2 >= 0.5).astype(int)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.
            Y (numpy.ndarray): Correct labels with shape (1, m), containing the
                              correct labels for the input data.
            A1 (numpy.ndarray): Output of the hidden layer with shape (nodes, m).
            A2 (numpy.ndarray): Predicted output with shape (1, m).
            alpha (float): The learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]

        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer gradients
        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights and biases
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
