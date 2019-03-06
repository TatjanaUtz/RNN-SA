"""Utility functions helpfull within project."""

from yaml import load
from tensorflow.contrib.training import HParams
import tensorflow as tf

class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        """Constructor."""
        super().__init__()
        self.dictionary = dict()
        with open(yaml_fn) as fp:
            for k, v in load(fp)[config_name].items():
                self.add_hparam(k, v)
                self.dictionary[k] = v

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


if __name__ == "__main__":
    hparams = YParams('hparams.yaml', 'large_hidden')
    print(hparams.num_hidden)  # print 1024