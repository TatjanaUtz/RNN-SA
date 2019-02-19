"""Module Dataset.

Class representing a Dataset for working with neural networks and tensorflow.
"""


class Dataset():
    """Class representing a dataset for neural networks.

    A dataset contains labeled samples. A dataset is divided into 3 parts:
        training -- dataset for training a neural network
        test -- dataset for testing a neural network
        evaluation -- dataset for evaluating different configurations of a neural network
    Every sample responds to a task-set. The label states, if the task-set was successfully ran.
    """

    def __init__(self):
        """Constructor of class Dataset."""
        self.training_set = []
        self.test_set = []
        self.evaluation_set = []
