#!/usr/bin/env python3
"""
Deep Neural Network module for binary classification.
"""

import numpy as np
import pickle


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

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m), where nx is the
                              number of input features and m is the number of
                              examples.

        Returns:
            tuple: A tuple containing:
                - A (numpy.ndarray): The output of the neural network with shape
                  (nodes_in_last_layer, m).
                - cache (dict): A dictionary containing all intermediary values
                  of the network.
        """
        self.__cache['A0'] = X
        A = X

        for l in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(l)]
            b = self.__weights['b{}'.format(l)]
            z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache['A{}'.format(l)] = A

        return A, self.__cache

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
        A, _ = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m), containing the
                              correct labels for the input data.
            cache (dict): A dictionary containing all intermediary values of the
                         network.
            alpha (float): The learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        A_L = cache['A{}'.format(self.__L)]

        # Output layer gradient
        dZ = A_L - Y

        # Backpropagate through all layers
        for l in range(self.__L, 0, -1):
            A_prev = cache['A{}'.format(l - 1)]
            W = self.__weights['W{}'.format(l)]
            b = self.__weights['b{}'.format(l)]

            # Calculate gradients
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Update weights and biases
            self.__weights['W{}'.format(l)] = W - alpha * dW
            self.__weights['b{}'.format(l)] = b - alpha * db

            # Calculate dZ for previous layer (if not the first layer)
            if l > 1:
                A_prev_activated = cache['A{}'.format(l - 1)]
                dZ = np.dot(W.T, dZ) * A_prev_activated * (1 - A_prev_activated)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the deep neural network.

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
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance object to a file in pickle format.

        Args:
            filename (str): The file to which the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a pickled DeepNeuralNetwork object.

        Args:
            filename (str): The file from which the object should be loaded.

        Returns:
            DeepNeuralNetwork: The loaded object, or None if filename doesn't
                              exist.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
