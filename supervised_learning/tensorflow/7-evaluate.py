#!/usr/bin/env python3
"""
Module for evaluating the output of a neural network.
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X (numpy.ndarray): The input data to evaluate.
        Y (numpy.ndarray): The one-hot labels for X.
        save_path (str): The location to load the model from.

    Returns:
        tuple: A tuple containing:
            - y_pred (numpy.ndarray): The network's prediction.
            - accuracy (float): The accuracy of the network.
            - loss (float): The loss of the network.
    """
    # Import the meta graph
    saver = tf.train.import_meta_graph(save_path + '.meta')
    
    # Get tensors from the graph's collection
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    y_pred = tf.get_collection('y_pred')[0]
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
    
    # Evaluate the model
    with tf.Session() as sess:
        # Restore the model
        saver.restore(sess, save_path)
        
        # Run evaluation
        pred, acc, cost = sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})
    
    return pred, acc, cost
