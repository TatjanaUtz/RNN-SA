"""Utility functions helpfull within project."""

import logging  # for logging
import os
import time # for getting the local time

import tensorflow as tf
from tensorflow.contrib.training import HParams
from yaml import load


class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        """Constructor."""
        super().__init__()
        self.dictionary = dict()
        with open(yaml_fn) as fp:
            for k, v in load(fp)[config_name].items():
                self.add_hparam(k, v)
                self.dictionary[k] = v


class Config(HParams):
    def __init__(self, yaml_fn, config_name):
        """Constructor."""
        super().__init__()
        self.dictionary = dict()
        with open(yaml_fn) as fp:
            for k, v in load(fp)[config_name].items():
                self.add_hparam(k, v)
                self.dictionary[k] = v

        # add tensorboard log dir
        tensorboard_log_dir = os.path.join("experiments",
                                           time.strftime("%Y-%m-%d\\", time.localtime()),
                                           config_name, "logs\\")
        self.add_hparam('tensorboard_log_dir', tensorboard_log_dir)
        self.dictionary['tensorboard_log_dir'] = tensorboard_log_dir

        # add checkpoint dir
        checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d\\", time.localtime()),
                                      config_name, "checkpoints\\")
        self.add_hparam('checkpoint_dir', checkpoint_dir)
        self.dictionary['checkpoint_dir'] = checkpoint_dir

        # create tensorboard and checkpoint dir
        create_dirs([tensorboard_log_dir, checkpoint_dir])


def variable_summaries(var):
    """ This helper function from the TensorFlow documentation adds a few operations to log the
    summaries.

    Args:
        var -- variable that should be summarized
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def create_dirs(dirs):
    """To create directories.

    This function creates the given directories if these directories are not found.

    Args:
        dirs -- a list of directories
    """
    # create logger
    logger = logging.getLogger('RNN-SA.util.create_dirs')

    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as exc:
        logger.error("Creating directories error: %", exc)


if __name__ == "__main__":
    hparams = YParams('hparams.yaml', 'large_hidden')
    print(hparams.num_hidden)  # print 1024
