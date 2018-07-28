from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import cv2
import time

# Self imports
from load_datasets import load_datasets


# globals
IMAGE_SIZE = 128
NUM_CORES = 4

tf.logging.set_verbosity(tf.logging.INFO)



def imgs_input_fn(filenames, labels, perform_shuffle=True, batch_size=10):
    def convert_to_img(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_png(image_string, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        resized_image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
        return  {'x': resized_image}, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if perform_shuffle:
        # should be very large number if the entire dataset can't be shuffled
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.repeat(None)

    dataset = dataset.map(map_func=convert_to_img, num_parallel_calls = NUM_CORES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset



def cnn_face_classification_fn(features, labels, mode, params):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # images are 128x128 pixels, and use gray scale color channel
    input_layer = tf.reshape(features['x'], [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    #print("input layer shape is {}".format(input_layer.get_shape()))

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

    #print('conv1 shape is {}'.format(conv1.get_shape()))

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 128, 128, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #print('pool1 shape is {}'.format(pool1.get_shape()))

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

    #print('conv2 shape is {}'.format(conv2.get_shape()))

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 64, 64, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #print('pool2 shape is {}'.format(pool2.get_shape()))

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 32 * 32 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

    #print('pool2_flat shape is {}'.format(pool2_flat.get_shape()))

    # Dense Layer #1
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 62 * 62 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=params.dense_layer_units, activation=tf.nn.leaky_relu)
    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(inputs=dense1, rate=params.dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    """
    # Dense Layer #2
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1024]
    dense2 = tf.layers.dense(inputs=dropout1, units=params.dense_layer_units, activation=tf.nn.leaky_relu)
    # Add dropout operation; 0.6 probability that element will be kept
    dropout2 = tf.layers.dropout(inputs=dense2, rate=params.dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
    """

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout1, units=2)

    #print('logits shape is {}'.format(logits.get_shape()))

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
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights = params.cross_entropy_weight)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        if params.optimizer == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate)
        elif params.optimizer == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def unison_shuffled_copies(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def main():
    start_time = time.time()
    print("Loading data...")
    train_dataset, valid_dataset, eval_dataset = load_datasets()
    print("Done loading")

    train_data, train_labels = train_dataset
    train_labels = np.asarray(train_labels, dtype=np.int32)
    valid_data, valid_labels = valid_dataset
    valid_labels = np.asarray(valid_labels, dtype=np.int32)

    config = tf.estimator.RunConfig(model_dir = "Models/gray_scale_6.0",
                                    keep_checkpoint_max = 25,
                                    save_checkpoints_steps = 1000
                                    )

    params = tf.contrib.training.HParams(
        learning_rate = 0.001,
        dense_layers = 1,
        dense_layer_units = 1024,
        dropout_rate = 0.3,
        optimizer = "Adam",
        cross_entropy_weight = 1.0,
        ##### Higher level #######
        color_channels = 1
    )

    # Create the Estimator
    face_classifier = tf.estimator.Estimator(model_fn=cnn_face_classification_fn, config = config, params = params)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    my_train = True
    # the old way
    if my_train:
        face_classifier.train(input_fn=lambda: imgs_input_fn(train_data, train_labels, perform_shuffle = True, batch_size = 25),
                                               hooks=[logging_hook],
                                               steps = 5000)

    train_results = face_classifier.evaluate(input_fn=lambda: imgs_input_fn(train_data, train_labels, perform_shuffle = False, batch_size = 25))
    valid_results = face_classifier.evaluate(input_fn=lambda: imgs_input_fn(valid_data, valid_labels, perform_shuffle = False, batch_size = 25))

    print("------------------------------------------")
    print(train_results)
    print(valid_results)


    """
    # the new way
    train_spec  = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(train_data,
                                                                        train_labels,
                                                                        perform_shuffle=True,
                                                                        batch_size=15),
                                                                        max_steps=50000,
                                                                        hooks=[logging_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(valid_data,
                                                                     valid_labels,
                                                                     perform_shuffle=False,
                                                                     batch_size=len(valid_data)),
                                                                     steps = 5000)


    tf.estimator.train_and_evaluate(face_classifier, train_spec, eval_spec)
    """
    print("--- %s seconds ---" % (time.time() - start_time))


main()
