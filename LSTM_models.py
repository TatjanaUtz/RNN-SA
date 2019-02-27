"""Model classes."""

import logging
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from logging_config import log_results


class Single_LSTM_Model:
    """Single LSTM Model."""

    def __init__(self, epochs=8, n_classes=1, hidden_dim=200, n_features=36, sequence_length=4,
                 batch_size=35, input_dim=9, num_cells=1):
        """Constructor."""

        # specify hyperparameters
        self.input_dim = input_dim  # dimension of each element of the sequence
        self.sequence_length = sequence_length  # length of each sequence
        self.hidden_dim = hidden_dim  # size of RNN hidden dimension = hidden state (both c and h)

        self.epochs = epochs  # number of iterations to run the data set through the model
        self.n_classes = n_classes  # number of classes (binary classification: 0 = not schedulable, 1 = schedulable)
        self.n_features = n_features  # number of features in the dataset
        self.batch_size = batch_size  # size of each batch of data that is feed into the model
        self.learning_rate = 0.001  # learning rate
        self.num_cells = 1  # number of LSTM cells

        # define shapes of weights and biases manually
        # random value of shape [rnn_size, n_classes] and [n_classes]
        # automatically do this: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
        self.layer = {'weights': tf.Variable(tf.random_normal([self.hidden_dim, self.n_classes])),
                      'bias': tf.Variable(tf.random_normal([self.n_classes]))}

        # create placeholders for input data and labels
        self.xplaceholder = tf.placeholder(tf.float32, [None, self.n_features])
        self.yplaceholder = tf.placeholder(tf.float32)

        # assign data to x as a sequence: split feature batch along vertical dimension (=1) into
        # sequence_length slices
        # each slice is an element of the sequence given as input to the LSTM layer
        # (shape of one element of  sequence: [batch_size, num_features / sequence_length])
        self.x = tf.split(self.xplaceholder, self.sequence_length, 1)

        # define the cost function:
        # sigmoid_cross_entropy_with_logits because of binary classification
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.model(),  # tensor of type float32 or float64
            labels=self.yplaceholder  # tensor of the same type and shape as logits
        ))

        # pass cost to the optimizer:
        # AdamOptimizer because of fairly better performance
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate  # the learning rate
        ).minimize(self.cost)

        # auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """Model function."""

        # create LSTM cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.hidden_dim,  # number of units in the LSTM cell
            use_peepholes=False,  # enable/disable diagonal/peephole connections
            cell_clip=None,
            # cell state is clipped by this value prior to the cell output activation
            initializer=None,  # initializer to use for weight and projection matrices
            num_proj=None,  # output dimensionality for the projection matrices
            proj_clip=None,
            # the projected values are clipped elementwise to within [-proj_clip, proj_clip]
            forget_bias=1.0,  # biases of the forget gate
            activation=None,  # activation function of the inner states
            reuse=None,  # whether to reuse variables in an existing scope
        )

        # run the cell on the input to obtain tensors for outputs and states
        # outputs = outputs of the LSTM layer for each time step
        # states = value of last state of both the hidden states (h and c)
        outputs, states = tf.nn.static_rnn(
            cell=lstm_cell,  # an instance of RNNCell
            inputs=self.x,
            # a length T list of inputs, each a tensor of shape [batch_size, input_size]
            initial_state=None,  # an initial state for the RNN
            dtype=tf.float32,  # data type for the initial state and expected output
            sequence_length=None,  # specifies the length of each sequence in inputs
            scope=None  # VariableScope for the created subgraph
        )
        # outputs, states = tf.nn.dynamic_rnn(
        #     cell=lstm_cell,  # an instance of RNNCell
        #     inputs=self.x,  # the RNN inputs
        #     sequence_length=None,  # an int vector sized [batch_size], used to copy-through state
        #     # and zero-out outputs when past a batch element's sequence length
        #     initial_state=None,  # an initial state for the RNN
        #     dtype=tf.float32,  # the data type for the initial state and expected output
        #     parallel_iterations=32,
        #     # number of iterations to run in parallel (trade-off between time and memory)
        #     swap_memory=False,  # transparently swap the tensors produced in forward inference but
        #     # needed for back prop from GPU to CPU
        #     time_major=False,  # shape format of the inputs and outputs tensors,
        #     # True = [max_time, batch_size, depth], False = [batch_size, max_time, depth]
        #     scope=None  # VariableScope for the created subgraph
        # )

        # take only the last output of the LSTM layer, multiply it with the previouly defined
        # weight matrix and add the bias value
        # result = logit value of forward propagation
        logit = tf.matmul(outputs[-1], self.layer['weights']) + self.layer['bias']

        # reshape matrix into vector (shape of labels and logits should be equal for feeding into
        # cost function)
        logit = tf.reshape(logit, [-1])

        # return reshaped logit value
        return logit

    def train(self, X_train, y_train):
        """Train function."""
        # create logger
        logger = logging.getLogger('RNN-SA.models.Single_LSTM_Model.train')

        with tf.Session() as sess:

            # initialize all global and local variables
            tf.get_variable_scope().reuse_variables()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            # loop over number of iterations (epochs)
            for epoch in range(self.epochs):
                # save start time
                start_time = time.time()

                # reset epoch loss to 0
                epoch_loss = 0

                # define variable to keep track of start and end computation when splitting data into batches
                i = 0

                # Loop over number of batches
                for step in range(int(len(X_train) / self.batch_size)):
                    # keep track from where data was split in each iteration
                    start = i
                    end = i + self.batch_size

                    # assign a batch of features and labels
                    batch_x = np.array(X_train[start:end])
                    batch_y = np.array(y_train[start:end])

                    # tell tensorflow to run the subgraph necessary to compute the optimizer and the
                    # cost by feeding the values in batch_x and batch_y to the placeholders
                    # compute value of optimizer and cost and assign them to the variables
                    _, c = sess.run([self.optimizer, self.cost],
                                    feed_dict={self.xplaceholder: batch_x,
                                               self.yplaceholder: batch_y})

                    # add loss of current batch to epoch_loss
                    epoch_loss += c

                    # raise iterator through data batches
                    i += self.batch_size

                # stop time
                stop_time = time.time()

                # print total loss of epoch
                logger.info("Epoch %d completed out of %d, loss = %f", epoch, self.epochs,
                            epoch_loss)
                logger.info("Time elapsed: %f", stop_time - start_time)

            # Save model
            self.save_path = os.path.join(os.getcwd(), 'single_lstm_model.ckpt')
            self.saver.save(sess, self.save_path)

    def test(self, X_test, y_test):
        """Test function."""
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()

            # Load saved model
            self.saver.restore(sess, self.save_path)

            # feed testing data set into the model and tell tensorflow to run the subgraph necessary
            # to compute logit
            # pass logit value through a sigmoid activation to get prediction
            # round off to remove decimal places of predicted values
            pred = tf.round(tf.nn.sigmoid(self.model())).eval(
                {self.xplaceholder: np.array(X_test),
                 self.yplaceholder: np.array(y_test)})

            # calculate F1 score = weighted average of precision and recall
            f1 = f1_score(np.array(y_test), pred, average='macro')

            # calculate accurarcy score
            accuracy = accuracy_score(np.array(y_test), pred)

            # calculate recall = ratio of correctly predicted positive observations to all positive observations
            recall = recall_score(y_true=np.array(y_test), y_pred=pred)

            # calculate precision = ratio of correctly predicted positive observations to total predicted positive observations
            precision = precision_score(y_true=np.array(y_test), y_pred=pred)

            # print out all calculated scores
            log_results("Single LSTM Model",
                        {"F1 Score": f1, "Accuracy Score": accuracy, "Recall": recall,
                         "Precision": precision})

class dynamic_Single_LSTM_Model:
    """Single LSTM Model."""

    def __init__(self, hparams):
        """Constructor."""

        # save hyperparameters
        self.hparams = hparams

        # define shapes of weights and biases manually
        # random value of shape [rnn_size, n_classes] and [n_classes]
        # automatically do this: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
        self.layer = {'weights': tf.Variable(tf.random_normal([self.hparams.hidden_dim, self.hparams.num_classes])),
                      'bias': tf.Variable(tf.random_normal([self.hparams.num_classes]))}

        # create placeholders for input data and labels
        self.xplaceholder = tf.placeholder(tf.float32, [None, self.hparams.time_steps, self.hparams.num_features])
        self.yplaceholder = tf.placeholder(tf.float32)

        # define the cost function:
        # sigmoid_cross_entropy_with_logits because of binary classification
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.model(),  # tensor of type float32 or float64
            labels=self.yplaceholder  # tensor of the same type and shape as logits
        ))

        # pass cost to the optimizer:
        # AdamOptimizer because of fairly better performance
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hparams.learning_rate  # the learning rate
        ).minimize(self.cost)

        # auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """Model function."""

        # create LSTM cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.hparams.hidden_dim,  # number of units in the LSTM cell
            use_peepholes=False,  # enable/disable diagonal/peephole connections
            cell_clip=None,
            # cell state is clipped by this value prior to the cell output activation
            initializer=None,  # initializer to use for weight and projection matrices
            num_proj=None,  # output dimensionality for the projection matrices
            proj_clip=None,
            # the projected values are clipped elementwise to within [-proj_clip, proj_clip]
            forget_bias=1.0,  # biases of the forget gate
            activation=None,  # activation function of the inner states
            reuse=None,  # whether to reuse variables in an existing scope
        )

        # run the cell on the input to obtain tensors for outputs and states
        # outputs = outputs of the LSTM layer for each time step
        # states = value of last state of both the hidden states (h and c)
        # outputs, states = tf.nn.static_rnn(
        #     cell=lstm_cell,  # an instance of RNNCell
        #     inputs=self.x,
        #     # a length T list of inputs, each a tensor of shape [batch_size, input_size]
        #     initial_state=None,  # an initial state for the RNN
        #     dtype=tf.float32,  # data type for the initial state and expected output
        #     sequence_length=None,  # specifies the length of each sequence in inputs
        #     scope=None  # VariableScope for the created subgraph
        # )
        outputs, states = tf.nn.dynamic_rnn(
            cell=lstm_cell,  # an instance of RNNCell
            inputs=self.xplaceholder,  # the RNN inputs
            sequence_length=None,  # an int vector sized [batch_size], used to copy-through state
            # and zero-out outputs when past a batch element's sequence length
            initial_state=None,  # an initial state for the RNN
            dtype=tf.float32,  # the data type for the initial state and expected output
            parallel_iterations=32,
            # number of iterations to run in parallel (trade-off between time and memory)
            swap_memory=False,  # transparently swap the tensors produced in forward inference but
            # needed for back prop from GPU to CPU
            time_major=False,  # shape format of the inputs and outputs tensors,
            # True = [max_time, batch_size, depth], False = [batch_size, max_time, depth]
            scope=None  # VariableScope for the created subgraph
        )

        # take only the last output of the LSTM layer
        # outputs[-1] isn't supported by tensorflow
        outputs = tf.transpose(outputs, [1, 0, 2])
        last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
        # last_output = outputs[-1]

        # multiply last output with the previously defined weight matrix and add the bias value
        # result = logit value of forward propagation
        logit = tf.matmul(last_output, self.layer['weights']) + self.layer['bias']

        # reshape matrix into vector (shape of labels and logits should be equal for feeding into
        # cost function)
        logit = tf.reshape(logit, [-1])

        # return reshaped logit value
        return logit

    def train(self, X_train, y_train):
        """Train function."""
        # create logger
        logger = logging.getLogger('RNN-SA.models.Single_LSTM_Model.train')

        with tf.Session() as sess:

            # initialize all global and local variables
            tf.get_variable_scope().reuse_variables()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            # loop over number of iterations (epochs)
            for epoch in range(self.hparams.num_epochs):
                # save start time
                start_time = time.time()

                # reset epoch loss to 0
                epoch_loss = 0

                # define variable to keep track of start and end computation when splitting data into batches
                i = 0

                # Loop over number of batches
                for step in range(int(len(X_train) / self.hparams.batch_size)):
                    # keep track from where data was split in each iteration
                    start = i
                    end = i + self.hparams.batch_size

                    # assign a batch of features and labels
                    batch_x = np.array(X_train[start:end])
                    batch_y = np.array(y_train[start:end])

                    # tell tensorflow to run the subgraph necessary to compute the optimizer and the
                    # cost by feeding the values in batch_x and batch_y to the placeholders
                    # compute value of optimizer and cost and assign them to the variables
                    _, c = sess.run([self.optimizer, self.cost],
                                    feed_dict={self.xplaceholder: batch_x,
                                               self.yplaceholder: batch_y})

                    # add loss of current batch to epoch_loss
                    epoch_loss += c

                    # raise iterator through data batches
                    i += self.hparams.batch_size

                # stop time
                stop_time = time.time()

                # print total loss of epoch
                logger.info("Epoch %d completed out of %d, loss = %f", epoch, self.hparams.num_epochs,
                            epoch_loss)
                logger.info("Time elapsed: %f", stop_time - start_time)

            # Save model
            self.save_path = os.path.join(os.getcwd(), 'single_lstm_model.ckpt')
            self.saver.save(sess, self.save_path)

    def test(self, X_test, y_test):
        """Test function."""
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()

            # Load saved model
            self.saver.restore(sess, self.save_path)

            # feed testing data set into the model and tell tensorflow to run the subgraph necessary
            # to compute logit
            # pass logit value through a sigmoid activation to get prediction
            # round off to remove decimal places of predicted values
            pred = tf.round(tf.nn.sigmoid(self.model())).eval(
                {self.xplaceholder: np.array(X_test),
                 self.yplaceholder: np.array(y_test)})

            # calculate F1 score = weighted average of precision and recall
            f1 = f1_score(np.array(y_test), pred, average='macro')

            # calculate accurarcy score
            accuracy = accuracy_score(np.array(y_test), pred)

            # calculate recall = ratio of correctly predicted positive observations to all positive observations
            recall = recall_score(y_true=np.array(y_test), y_pred=pred)

            # calculate precision = ratio of correctly predicted positive observations to total predicted positive observations
            precision = precision_score(y_true=np.array(y_test), y_pred=pred)

            # print out all calculated scores
            log_results("dynamic Single LSTM Model",
                        {"F1 Score": f1, "Accuracy Score": accuracy, "Recall": recall,
                         "Precision": precision})



if __name__ == "__main__":
    print("Main function von LSTM_models.py")
