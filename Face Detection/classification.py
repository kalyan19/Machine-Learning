from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import cv2

# Self imports
from load_datasets import load_datasets


# globals
IMAGE_SIZE = 128

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_face_classification_fn(features, labels, mode):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # images are 128x128 pixels, and use gray scale color channel
    input_layer = tf.reshape(features["x"], [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 128, 128, 1]
    # Output Tensor Shape: [batch_size, 128, 128, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    print('conv1 shape is {}'.format(conv1.get_shape()))

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 128, 128, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    print('pool1 shape is {}'.format(pool1.get_shape()))

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 64]
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

    print('conv2 shape is {}'.format(conv2.get_shape()))

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 64, 64, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    print('pool2 shape is {}'.format(pool2.get_shape()))

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 32 * 32 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

    print('pool2_flat shape is {}'.format(pool2_flat.get_shape()))

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 62 * 62 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    print('dense shape is {}'.format(dense.get_shape()))

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout, units=2)

    print('logits shape is {}'.format(logits.get_shape()))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    print("Loading data...")
    train_dataset, valid_dataset, eval_dataset = load_datasets()
    print("Done loading")

    train_data, train_labels = train_dataset
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data, eval_labels = eval_dataset
    eval_labels = np.asarray(eval_labels, dtype=np.int32)
    print(train_data.shape)
    print(train_labels.shape)
    print(train_data.dtype)
    print(train_labels.dtype)

    # Create the Estimator
    face_classifier = tf.estimator.Estimator(model_fn=cnn_face_classification_fn, model_dir="/tmp/face_class_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=200)

    """
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    face_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    """

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = face_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    """
      # Load training and eval data
      mnist = tf.contrib.learn.datasets.load_dataset("mnist")
      train_data = mnist.train.images  # Returns np.array
      train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
      eval_data = mnist.test.images  # Returns np.array
      eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    """



main()
