#!/usr/bin/env python3
"""
Module for building, training, and saving a neural network classifier.
"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train (numpy.ndarray): The training input data.
        Y_train (numpy.ndarray): The training labels.
        X_valid (numpy.ndarray): The validation input data.
        Y_valid (numpy.ndarray): The validation labels.
        layer_sizes (list): A list containing the number of nodes in each layer
                           of the network.
        activations (list): A list containing the activation functions for each
                           layer of the network.
        alpha (float): The learning rate.
        iterations (int): The number of iterations to train over.
        save_path (str): The path where to save the model. Defaults to
                        "/tmp/model.ckpt".

    Returns:
        str: The path where the model was saved.
    """
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    
    # Create forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    
    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    
    # Create training operation
    train_op = create_train_op(loss, alpha)
    
    # Add to graph collections
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Create saver
    saver = tf.train.Saver()
    
    # Start training session
    with tf.Session() as sess:
        sess.run(init)
        
        # Print at iteration 0 (before training)
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        
        print("After {} iterations:".format(0))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_acc))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_acc))
        
        # Train for specified iterations
        for i in range(1, iterations + 1):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            
            # Print every 100 iterations, and at the final iteration
            if i % 100 == 0 or i == iterations:
                train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
                valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
                
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))
        
        # Save the model
        save_path = saver.save(sess, save_path)
    
    return save_path
