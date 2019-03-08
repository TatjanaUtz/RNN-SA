"""An Example of a custom Estimator.
source: TensorFlow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import logging_config
from database_interface import Database
from util import YParams

# TASK_ATTRIBUTES = ['priority', 'deadline', 'quota', 'caps', 'pkg', 'arg', 'cores', 'coreoffset',
#                    'criticaltime', 'period', 'numberOfJobs', 'offset']
TASK_ATTRIBUTES = ["Priority", "PKG", "Arg", "CRITICALTIME", "Period", "Number_of_Jobs"]

TASK_PKGS = ['cond_mod', 'hey', 'pi', 'tumatmul']


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for training."""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def define_feature_columns():
    """Function to define the feature columns.

    Return:
        list with defined feature columns
    """
    priority = tf.feature_column.numeric_column(key='priority', dtype=tf.int64)
    # deadline = tf.feature_column.numeric_column(key='deadline', dtype=tf.int64)
    # quota = tf.feature_column.numeric_column(key='quota', dtype=tf.int64)
    # caps = tf.feature_column.numeric_column(key='caps', dtype=tf.int64)
    pkg = tf.feature_column.categorical_column_with_vocabulary_list(key='pkg',
                                                                    vocabulary_list=TASK_PKGS)
    arg = tf.feature_column.numeric_column(key='arg', dtype=tf.int64)
    # cores = tf.feature_column.numeric_column(key='cores', dtype=tf.int64)
    # coreoffset = tf.feature_column.numeric_column(key='coreoffset', dtype=tf.int64)
    criticaltime = tf.feature_column.numeric_column(key='criticaltime', dtype=tf.int64)
    period = tf.feature_column.numeric_column(key='period', dtype=tf.int64)
    numberOfJobs = tf.feature_column.numeric_column(key='numberOfJobs', dtype=tf.int64)
    # offset = tf.feature_column.numeric_column(key='offset', dtype=tf.int64)

    # return list with defined feature columns
    # return [priority, deadline, quota, caps, pkg, arg, cores, coreoffset, criticaltime, period,
    #         numberOfJobs, offset]
    return [priority, pkg, arg, criticaltime, period, numberOfJobs]


def model_fn(features, labels, mode, params):
    """Model function for RNN classifier.

    The input is given to LSTM layers (as configured with params.num_layers and params.num_nodes).
    The final state of the all LSTM layers are concatenated and fed to a fully connected layer to
    obtain the final classification scores.

    Args:
        features -- batches of features returned from the input function
        labels -- batches of labels returned from the input function
        mode -- an instance of tf.estimator.ModeKeys (training, predicting, or evaluation)
        params -- additional configuration
    """
    # use 'input_layer' to apply the feature columns
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # TODO: Use RNN instead
    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # create training op
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    # initialize logging
    logging_config.init_logging()

    # get hyperparameters
    hparams = YParams('hparams.yaml', 'LSTM')

    my_database = Database()
    features, labels, _ = my_database.read_all_tasksets_preprocessed()

    X_train, X_test, y_train, y_test = train_test_split(features, labels)

    # Example with keras
    # create the model
    model = Sequential()
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    print(model.summary())

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # # instantiate an estimator, passing the feature columns
    # my_feature_columns = define_feature_columns()
    #
    # classifier = tf.estimator.Estimator(
    #     model_fn=model_fn,
    #     params={
    #         'feature_columns': my_feature_columns,
    #         'hidden_units': [10, 10],
    #         'n_classes': 2,
    #     })
    #
    # # Train the Model
    # classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, hparams.batch_size),
    #                  steps=hparams.num_epochs)
    #
    # # evaluate the model
    # eval_result = classifier.evaluate(
    #     input_fn=lambda: eval_input_fn(test_x, test_y, hparams.batch_size))
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
