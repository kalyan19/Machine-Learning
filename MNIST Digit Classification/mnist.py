from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # load the data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # None means that the dimension can be of any length (any num of 784-d vectors)
    x = tf.placeholder(tf.float32, [None, 784])

    # these are tf.Variables since TF changes them internally
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # predicted output to given input (without applying the activation function yet)
    y_pred = tf.matmul(x, W) + b
    # true output to associated input
    y_true = tf.placeholder(tf.float32, [None, 10])

    # the cost function (softmax + cross entropy can be combined for optimization)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred)

    # learning algorithm
    learning_rate = 0.5
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # model can only be run by creating a session
    sess = tf.InteractiveSession()

    # initializes all tf variable stuff
    tf.global_variables_initializer().run()

    # train the model
    epochs = 1000
    batch_size = 100
    for _ in range(epochs):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # compares if the prediction is correct and returns true/false
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))

    # converts the booleans to ints and takes average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # run test data and check accuracy
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

main()
