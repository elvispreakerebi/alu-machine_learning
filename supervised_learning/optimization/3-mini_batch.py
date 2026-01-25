#!/usr/bin/env python3
"""
Module for training a neural network using mini-batch gradient descent.
"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train (numpy.ndarray): Training data of shape (m, 784), where m is the
                                number of data points.
        Y_train (numpy.ndarray): Training labels (one-hot) of shape (m, 10),
                                 where 10 is the number of classes.
        X_valid (numpy.ndarray): Validation data of shape (m, 784).
        Y_valid (numpy.ndarray): Validation labels (one-hot) of shape (m, 10).
        batch_size (int): The number of data points in a batch. Defaults to 32.
        epochs (int): The number of times the training should pass through the
                     whole dataset. Defaults to 5.
        load_path (str): The path from which to load the model. Defaults to
                        "/tmp/model.ckpt".
        save_path (str): The path to where the model should be saved after
                        training. Defaults to "/tmp/model.ckpt".

    Returns:
        str: The path where the model was saved.
    """
    # Import meta graph and restore session
    saver = tf.train.import_meta_graph(load_path + '.meta')
    
    # Get tensors and ops from the collection
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    accuracy = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]
    
    # Start session
    with tf.Session() as sess:
        # Restore the model
        saver.restore(sess, load_path)
        
        # Print metrics before training (epoch 0)
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        
        print("After 0 epochs:")
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_acc))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_acc))
        
        # Loop over epochs
        for epoch in range(1, epochs + 1):
            # Shuffle data
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, Y_train)
            
            # Get the number of batches
            m = X_train.shape[0]
            num_batches = (m + batch_size - 1) // batch_size  # Ceiling division
            
            # Loop over batches
            step_number = 0
            for i in range(0, m, batch_size):
                # Get batch
                X_batch = X_train_shuffled[i:i + batch_size]
                Y_batch = Y_train_shuffled[i:i + batch_size]
                
                # Train on batch
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step_number += 1
                
                # Print every 100 steps
                if step_number % 100 == 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_acc = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step_number))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))
            
            # Print metrics after epoch
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))
        
        # Save the model
        save_path = saver.save(sess, save_path)
    
    return save_path
