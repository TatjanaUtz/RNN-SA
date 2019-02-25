"""Classes for representation of a dataset.

A dataset consists of many samples, that each hold a sequence of attributes.
All samples are divided into sub-datasets for training, test and evaluation.
"""
import numpy as np

TRAIN_PERCENTAGE = 0.8  # how much data should be used for training (in %)
TEST_PERCENTAGE = 0.1  # how much data should be used for testing (in %)


class Dataset():
    """Represents a dataset.
    A dataset consists of task-sets and the corresponding labels.
    """

    def __init__(self, input_data, labels):
        """Constructor of class Dataset.

        Args:
            input_data -- array with input data
            labels -- array with labels
        """
        self.input_data = input_data
        self.labels = labels

        # Split dataset for training, test and evaluation
        num_samples = len(input_data)  # lenght of dataset (number of samples)
        num_train = int(TRAIN_PERCENTAGE * num_samples)  # number of samples for training
        num_test = int(TEST_PERCENTAGE * num_samples)  # number of samples for testing

        # dataset for training
        self.train = Datatype('train', input_data[:num_train], labels[:num_train], num_train)

        # dataset for test
        self.test = Datatype('test', input_data[num_train:num_train + num_test],
                             labels[num_train:num_train + num_test], num_test)

        # dataset for evaluation
        self.eval = Datatype('eval', input_data[num_train + num_test:],
                             labels[num_train + num_test:], num_samples - num_train - num_test)


class Datatype():
    """Representation of a specific dataset for training, test or evaluation."""

    def __init__(self, name, input_data, labels, num_samples):
        """Constructor of class Datatype."""
        self.name = name  # name of the dataset
        self.input = input_data  # the input data
        self.labels = labels  # the labels
        self.num_samples = num_samples  # the number of samples
        self.global_count = 0  # number of samples that have already been used

    def next_batch(self, batch_size):
        """Get next batch of a dataset.

        Args:
            data_type -- dataset (train, test or eval) for which the next batch should be created
            batch_size -- size of the batch
        Return:
            input_data -- batch of input data
            labels -- batch of labels
        """
        # get count of dataset (= how much data has already been used)
        count = self.global_count

        # create next batches for input data and labels
        # TODO: replace sample_length with constant
        input_data, new_count = self.make_batch(self.input, batch_size, 36, count)
        labels, _ = self.make_batch(self.labels, batch_size, 1, count)

        # raise count of dataset
        self.global_count = new_count % self.num_samples

        # return input data and labels
        return input_data, labels

    @staticmethod
    def make_batch(data, batch_size, sample_length, count):
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
        # TODO: uncomment this block if lists are used
        # while len(batch) < batch_size:
        #     batch.append(np.zeros(sample_length, dtype=int))
        #     count = 0  # reset count (all data has been used)

        # TODO: delete this block if lists are used
        while len(batch) < batch_size:
            zeroes_np = np.asarray([[0] * sample_length])
            batch = np.append(batch, zeroes_np, axis=0)
            count = 0


        # return created batch and new count
        return batch, count


