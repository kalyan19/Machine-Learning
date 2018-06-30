from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data


def create_weights(shape, std = 0.01):
    shape = tf.TensorShape(shape)
    init_values = tf.truncated_normal(shape, stddev = std)
    return tf.Variable(init_values)

def create_biases(shape):
    shape = tf.TensorShape(shape)
    init_values = tf.zeros(shape)
    return tf.Variable(init_values)

# input -> hidden layer -> output
def build_network(input_layer, output_layer_size, neurons = 30):
    # hidden layer
    w1 = create_weights([input_layer.shape[1], neurons])
    b1 = create_biases([neurons])
    z1 = tf.add(tf.matmul(input_layer, w1), b1) # z1 = x * w + b
    a1 = tf.nn.sigmoid(z1) # a = sigmoid(z)
    # output layer
    w2 = create_weights([neurons, output_layer_size])
    b2 = create_biases([output_layer_size])
    z2 = tf.add(tf.matmul(a1, w2), b2)
    output_layer = tf.nn.sigmoid(z2)
    return output_layer

# input -> output
def build_simple_network(input_layer, output_layer_size):
    w1 = create_weights([input_layer.shape[1], output_layer_size])
    b1 = create_biases([output_layer_size])
    z1 = tf.add(tf.matmul(input_layer, w1), b1) # z1 = x * w + b
    return z1

def create_cost_fn(y_pred, y_true):
    # change this later to logits
    #cost = tf.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred))
    cost = tf.losses.mean_squared_error(y_true, y_pred)
    return cost

def create_optimizer(cost, learning_rate = 30):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return optimizer

def train(optimizer, mnist_data, x, y_true, sess, epochs = 1000, batch_size = 100):
    for _ in range(epochs):
      batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
      sess.run(optimizer, feed_dict={x: batch_xs, y_true: batch_ys})

def calc_accuracy(x, y_pred, y_true, mnist_data, sess):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist_data.test.images, y_true: mnist_data.test.labels}))

def setup_session():
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    return sess

def main():
    # load the data
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # set input, and true output
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])

    y_pred = build_network(x, 10)
    cost = create_cost_fn(y_pred, y_true)
    optimizer = create_optimizer(cost)
    sess = setup_session()
    train(optimizer, mnist_data, x, y_true, sess)

    calc_accuracy(x, y_pred, y_true, mnist_data, sess)



main()
