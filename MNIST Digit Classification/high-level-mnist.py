from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data


def model_fn(features, labels, mode):
    x = features['x']

    weights = tf.initializers.truncated_normal(stddev = 0.1)
    neurons = 30
    hidden_layer = tf.layers.dense(inputs = x, units = neurons, activation = tf.nn.sigmoid, kernel_initializer = weights)
    output_layer_size = 10
    logits = tf.layers.dense(inputs = hidden_layer, units = output_layer_size, kernel_initializer = weights)
    # """activation = tf.nn.softmax,"""
    predictions = {
        'class': tf.argmax(input = logits, axis=1),
        'probabilities':  tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #loss = tf.losses.mean_squared_error(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["class"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    classifier = tf.estimator.Estimator(model_fn=model_fn)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    classifier.train(input_fn = train_input_fn, steps = 20000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)


main()
