"""Classes for representation of a dataset.

A dataset consists of many samples, that each hold a sequence of attributes.
All samples are divided into sub-datasets for training, test and evaluation.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from database_interface import Database


TRAIN_PERCENTAGE = 0.8  # how much data should be used for training (in %)
TEST_PERCENTAGE = 0.2  # how much data should be used for testing (in %)


class Dataset():
    """Represents a dataset.
    A dataset consists of task-sets and the corresponding labels.
    """

    def __init__(self, hparams):
        """Constructor of class Dataset."""
        db = Database()
        input_data, labels, sequence_lengths = db.read_all_tasksets_preprocessed()

        # Split dataset for training, test and evaluation
        num_samples = len(input_data)  # lenght of dataset (number of samples)
        num_train = int(hparams.train_size * num_samples)  # number of samples for training
        num_test = int(hparams.test_size * num_samples)  # number of samples for testing

        # dataset for training
        self.train = Datatype('train', input_data[:num_train],
                              labels[:num_train],
                              sequence_lengths[:num_train],
                              num_train, hparams)

        # dataset for test
        self.test = Datatype('test', input_data[num_train:num_train + num_test],
                             labels[num_train:num_train + num_test],
                             sequence_lengths[num_train:num_train + num_test],
                             num_test, hparams)

        # dataset for evaluation
        self.eval = Datatype('eval', input_data[num_train + num_test:],
                             labels[num_train + num_test:],
                             sequence_lengths[num_train + num_test:],
                             num_samples - num_train - num_test, hparams)


class Datatype():
    """Representation of a specific dataset for training, test or evaluation."""

    def __init__(self, name, input_data, labels, sequence_lengths, num_samples, hparams):
        """Constructor of class Datatype."""
        self.name = name  # name of the dataset
        self.input = input_data  # the input data
        self.labels = labels  # the labels
        self.seqlen = sequence_lengths  # length of the sequences
        self.num_samples = num_samples  # the number of samples
        self.global_count = 0  # number of samples that have already been used
        self.hparams = hparams

    def next_batch(self):
        """Get next batch of a dataset.

        Args:
            batch_size -- size of the batch
        Return:
            input_data -- batch of input data
            labels -- batch of labels
        """
        # get count of dataset (= how much data has already been used)
        count = self.global_count

        # create next batches for input data and labels
        batch_x, new_count = self.make_batch(self.input, self.hparams.batch_size, count, [self.hparams.time_steps, self.hparams.element_size])
        batch_y, _ = self.make_batch(self.labels, self.hparams.batch_size, count, [None, 1])
        batch_seqlen, _ = self.make_batch(self.seqlen, self.hparams.batch_size, count)

        # raise count of dataset
        self.global_count = new_count % self.num_samples

        # return input data and labels
        return batch_x, batch_y, batch_seqlen

    @staticmethod
    def make_batch(data, batch_size, count, input_dim=None):
        """Create a batch.

        Args:
            data -- data array
            batch_size -- size of the batch
            sample_length -- size of one sample
            count -- count of data (= how much data was already used)
        Return:
            batch -- a batch of data
            count -- new count of this data array
        """
        batch = data[count:count + batch_size]  # create a batch of data starting at count
        count += batch_size  # raise count

        # if available data is less than batch_size, fill with zeros
        while len(batch) < batch_size:
            if input_dim is None:   # sequence length batch
                zeroes_np = np.asarray([0])
            elif input_dim[0] is None:  # label batch
                zeroes_np = np.asarray([[0] * input_dim[1]])
            else:   # input data batch
                zeroes_np = np.asarray([[[0] * input_dim[1]] * input_dim[0]])

            batch = np.append(batch, zeroes_np, axis=0)
            count = 0

        # return created batch and new count
        return batch, count


