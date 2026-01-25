#!/usr/bin/env python3
"""
Module for building, training, and saving neural network models with batch normalization.
"""

import numpy as np
import tensorflow as tf
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, 
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, 
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in TensorFlow using Adam 
    optimization, mini-batch gradient descent, learning rate decay, and batch normalization.

    Args:
        Data_train (tuple): Tuple containing (training inputs, training labels).
        Data_valid (tuple): Tuple containing (validation inputs, validation labels).
        layers (list): List containing the number of nodes in each layer.
        activations (list): List containing activation functions for each layer.
        alpha (float): Learning rate. Defaults to 0.001.
        beta1 (float): Weight for first moment of Adam. Defaults to 0.9.
        beta2 (float): Weight for second moment of Adam. Defaults to 0.999.
        epsilon (float): Small number to avoid division by zero. Defaults to 1e-8.
        decay_rate (float): Decay rate for inverse time decay. Defaults to 1.
        batch_size (int): Number of data points per mini-batch. Defaults to 32.
        epochs (int): Number of training epochs. Defaults to 5.
        save_path (str): Path to save the model. Defaults to '/tmp/model.ckpt'.

    Returns:
        str: The path where the model was saved.
    """
    # Extract training and validation data
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    
    # Get input dimensions
    m, nx = X_train.shape
    classes = Y_train.shape[1]
    
    # Create placeholders
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    
    # Build the network layer by layer
    prev = x
    for i in range(len(layers)):
        n = layers[i]
        activation = activations[i]
        
        if i == len(layers) - 1:
            # Last layer: output layer
            if activation is None:
                # No activation, just dense layer (produces logits)
                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
                logits = tf.layers.Dense(
                    units=n,
                    kernel_initializer=kernel_initializer,
                    name='output'
                )(prev)
                y_pred = logits
            else:
                # Has activation, use batch normalization
                # We need logits for loss, so build batch norm without activation first
                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
                dense_layer = tf.layers.Dense(
                    units=n,
                    kernel_initializer=kernel_initializer,
                    use_bias=False
                )
                Z = dense_layer(prev)
                # Apply batch normalization
                gamma = tf.Variable(tf.ones([n]), trainable=True)
                beta = tf.Variable(tf.zeros([n]), trainable=True)
                mean, variance = tf.nn.moments(Z, axes=[0], keep_dims=True)
                epsilon_bn = 1e-8
                Z_norm = tf.divide(tf.subtract(Z, mean), 
                                  tf.sqrt(tf.add(variance, epsilon_bn)))
                gamma_reshaped = tf.reshape(gamma, [1, n])
                beta_reshaped = tf.reshape(beta, [1, n])
                Z_bn = tf.add(tf.multiply(Z_norm, gamma_reshaped), beta_reshaped)
                logits = Z_bn
                # Apply activation
                y_pred = activation(Z_bn)
        else:
            # Hidden layers: use batch normalization
            prev = create_batch_norm_layer(prev, n, activation)
    
    # Calculate loss (softmax cross entropy)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=logits))
    
    # Calculate accuracy
    # Apply softmax to logits to get probabilities
    y_pred_probs = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(y_pred_probs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Create global_step variable for tracking training steps
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # Create epoch variable for learning rate decay (decay_step = 1 means per epoch)
    epoch_var = tf.Variable(0, trainable=False, name='epoch')
    
    # Create learning rate with decay based on epoch (decay_step = 1)
    alpha_decayed = learning_rate_decay(alpha, decay_rate, epoch_var, 1)
    
    # Create Adam optimizer
    train_op = tf.train.AdamOptimizer(
        learning_rate=alpha_decayed,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss, global_step=global_step)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Create saver
    saver = tf.train.Saver()
    
    # Training loop
    with tf.Session() as sess:
        sess.run(init)
        
        # Print metrics before first epoch (epoch 0)
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        
        print("After {} epochs:".format(0))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            # Set epoch variable for learning rate decay (learning rate remains constant within epoch)
            sess.run(tf.assign(epoch_var, epoch))
            
            # Shuffle training data before each epoch
            shuffle_indices = np.random.permutation(m)
            X_train_shuffled = X_train[shuffle_indices]
            Y_train_shuffled = Y_train[shuffle_indices]
            
            # Calculate number of batches
            num_batches = m // batch_size
            if m % batch_size != 0:
                num_batches += 1
            
            # Train on mini-batches
            step_in_epoch = 0
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, m)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                Y_batch = Y_train_shuffled[start_idx:end_idx]
                
                # Train on batch
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step_in_epoch += 1
                
                # Print every 100 steps
                if step_in_epoch % 100 == 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step_in_epoch))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
            
            # Print metrics after epoch
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            
            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
        
        # Save the model
        save_path_final = saver.save(sess, save_path)
    
    return save_path_final
