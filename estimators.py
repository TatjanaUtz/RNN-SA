"""Estimators for RNNs."""
import tensorflow as tf

from database import Database
from dataset import Dataset

TASK_ATTRIBUTES = ['priority', 'deadline', 'quota', 'caps', 'pkg', 'arg', 'cores', 'coreoffset',
                   'criticaltime', 'period', 'numberOfJobs', 'offset']
TASK_PKGS = ['cond_mod', 'hey', 'pi', 'tumatmul']


def train_input_fn(features, labels, batch_size, num_epochs):
    """Import function for training."""
    # convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # shuffle, repeat and batch the examples
    return dataset.shuffle(1000).repeat(count=num_epochs).batch(batch_size)


def eval_input_fn(dataset):
    """Import function for evaluation."""
    # manipulate dataset, extracting the feature dict and the label
    return feature_dict, label


def test_input_fn(dataset):
    """Import function for test."""
    # manipulate dataset, extracting the feature dict and the label
    return feature_dict, label


def define_feature_columns():
    """Function to define the feature columns.

    Return:
        list with defined feature columns
    """
    priority = tf.feature_column.numeric_column(key='priority', dtype=tf.int64)
    deadline = tf.feature_column.numeric_column(key='deadline', dtype=tf.int64)
    quota = tf.feature_column.numeric_column(key='quota', dtype=tf.int64)
    caps = tf.feature_column.numeric_column(key='caps', dtype=tf.int64)
    pkg = tf.feature_column.categorical_column_with_vocabulary_list(key='pkg',
                                                                    vocabulary_list=TASK_PKGS)
    arg = tf.feature_column.numeric_column(key='arg', dtype=tf.int64)
    cores = tf.feature_column.numeric_column(key='cores', dtype=tf.int64)
    coreoffset = tf.feature_column.numeric_column(key='coreoffset', dtype=tf.int64)
    criticaltime = tf.feature_column.numeric_column(key='criticaltime', dtype=tf.int64)
    period = tf.feature_column.numeric_column(key='period', dtype=tf.int64)
    numberOfJobs = tf.feature_column.numeric_column(key='numberOfJobs', dtype=tf.int64)
    offset = tf.feature_column.numeric_column(key='offset', dtype=tf.int64)

    # return list with defined feature columns
    return [priority, deadline, quota, caps, pkg, arg, cores, coreoffset, criticaltime, period,
            numberOfJobs, offset]


def my_model_fn(features, labels, mode, params):
    """Model function.

    Args:
        features -- batches of features returned from the input function
        labels -- batches of labels returned from the input function
        mode -- an instance of tf.estimator.ModeKeys (training, predicting, or evaluation)
        params -- additional configuration
    """
    # use 'input_layer' to apply the feature columns
    net = tf.feature_column.input_layer(features, params['feature_columns'])

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

    # Compute evaluation.
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

    # compute training.
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    # instantiate an estimator, passing the feature columns
    feature_columns = define_feature_columns()
    estimator = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })

    # call a training, evaluation or inference method
    estimator.train(input_fn=lambda: train_input_fn, steps=2000)

    eval_result = estimator.evaluate(input_fn=lambda: eval_input_fn())
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # import data to memory
    my_database = Database()
    tasksets, labels = my_database.read_all_tasksets()
    my_dataset = Dataset(tasksets, labels)

    # pass this data to input function
    batch_size = 100
    train_input_fn(my_dataset.train.input, my_dataset.train.labels, batch_size)
