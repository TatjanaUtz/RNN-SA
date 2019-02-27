"""Model classes."""

import logging
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from logging_config import log_results

TB_LOG_PATH = os.path.join(os.getcwd(), 'log')
SAVE_PATH = os.path.join(os.getcwd(), 'lstm_model.ckpt')


class LSTM_Model:
    """Single LSTM Model."""

    def __init__(self, hparams):
        """Constructor."""

        # save hyperparameters
        self.hparams = hparams

        # create placeholders for input data and labels
        self.xplaceholder = tf.placeholder(tf.float32, [None, self.hparams.time_steps,
                                                        self.hparams.num_features], name='inputs')
        self.yplaceholder = tf.placeholder(tf.float32, name='labels')

        # create model
        self.model()

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

        # add dropout layer if needed
        if self.hparams.keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=self.hparams.keep_prob)

        # stack cells into multiple layers if needed
        self.cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell for _ in range(self.hparams.num_cells)]
        ) if self.hparams.num_cells > 1 else lstm_cell

        # getting an inital state of all zeros
        self.initial_state = self.cell.zero_state(self.hparams.batch_size, tf.float32)

        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
            cell=self.cell,  # an instance of RNNCell
            inputs=self.xplaceholder,  # the RNN inputs
            sequence_length=None,  # an int vector sized [batch_size], used to copy-through state
            # and zero-out outputs when past a batch element's sequence length
            initial_state=self.initial_state,  # an initial state for the RNN
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
        # (Note: outputs[-1] isn't supported by tensorflow)
        lstm_outputs = tf.transpose(self.lstm_outputs, [1, 0, 2])
        last_output = tf.gather(lstm_outputs, int(lstm_outputs.get_shape()[0]) - 1,
                                name="last_lstm_output")

        # process output through a fully connected output layer
        # result = logit value of forward propagation
        self.logits = tf.contrib.layers.fully_connected(inputs=last_output,
                                                        num_outputs=self.hparams.num_classes)

        # reshape matrix into vector (shape of labels and logits should be equal for feeding into
        # loss function)
        self.logits = tf.reshape(self.logits, [-1], "lstm_logit")

        # define the loss function:
        # sigmoid_cross_entropy_with_logits because of binary classification
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.yplaceholder
        ), name="loss_sigmoid_cross_entropy")

        # pass loss to the optimizer: AdamOptimizer because of fairly better performance
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hparams.learning_rate  # the learning rate
        ).minimize(self.loss, name="loss_sigmoid_cross_entropy_adam_minimize")

        correct_pred = tf.equal(tf.cast(tf.round(self.logits), tf.float32), self.yplaceholder)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # return reshaped logit value
        # return logit

    def train(self, X_train, y_train):
        """Train function."""
        # create logger
        logger = logging.getLogger('RNN-SA.models.Single_LSTM_Model.train')

        with tf.Session() as sess:
            # initialize all variables
            sess.run(tf.global_variables_initializer())

            # calculate number of batches
            num_batches = len(X_train) // self.hparams.batch_size

            # loop over number of iterations (epochs)
            for epoch in range(self.hparams.num_epochs):
                # save start time
                start_time = time.time()

                # create initial state
                state = sess.run(self.initial_state)

                # reset epoch loss to 0
                epoch_loss = 0

                # reset training accuracy
                train_acc = []

                # define variable to keep track of start and end computation when splitting data into batches
                i = 0

                # Loop over number of batches
                for step in range(int(num_batches)):
                    # keep track from where data was split in each iteration
                    start = i
                    end = i + self.hparams.batch_size

                    # assign a batch of features and labels
                    batch_x = np.array(X_train[start:end])
                    batch_y = np.array(y_train[start:end])

                    # tell tensorflow to run the subgraph necessary to compute the optimizer and the
                    # cost by feeding the values in batch_x and batch_y to the placeholders
                    # compute value of optimizer and cost and assign them to the variables
                    # _, c = sess.run([self.optimizer, self.loss],
                    #                 feed_dict={self.xplaceholder: batch_x,
                    #                            self.yplaceholder: batch_y})
                    loss, state, _, batch_acc = sess.run(
                        [self.loss, self.final_state, self.optimizer, self.accuracy],
                        feed_dict={self.xplaceholder: batch_x,
                                   self.yplaceholder: batch_y})

                    # add loss of current batch to epoch_loss
                    epoch_loss += loss

                    # add training accuracy of current batch to train_acc
                    train_acc.append(batch_acc)

                    # raise iterator through data batches
                    i += self.hparams.batch_size

                    # check if last batch of this epoch was processed
                    if (step + 1) % num_batches == 0:
                        logger.info("Epoch: %d/%d... Batch: %d/%d..., Train Loss: %.3f... "
                                    "Train Accuracy: %.3f", epoch + 1, self.hparams.num_epochs,
                                    step + 1, num_batches, loss, np.mean(train_acc))

                # stop time
                stop_time = time.time()

                # print total loss of epoch
                logger.info("Epoch %d completed out of %d, loss = %f", epoch,
                            self.hparams.num_epochs,
                            epoch_loss)
                logger.info("Time elapsed: %f", stop_time - start_time)

            # Save model
            self.saver.save(sess, SAVE_PATH)

    def test(self, X_test, y_test):
        """Test function."""
        test_acc = []
        test_pred = []
        num_batches = len(X_test) // self.hparams.batch_size

        with tf.Session() as sess:
            # Restore saved model
            self.saver.restore(sess, SAVE_PATH)

            test_state = sess.run(self.cell.zero_state(len(X_test), tf.float32))

            # loss, _, batch_acc = sess.run([self.loss, self.optimizer, self.accuracy],
            #                               feed_dict={self.xplaceholder: np.array(X_test),
            #                                          self.yplaceholder: np.array(y_test)})

            # define variable to keep track of start and end computation when splitting data into batches
            i = 0

            # Loop over number of batches
            for step in range(int(num_batches)):
                # keep track from where data was split in each iteration
                start = i
                end = i + self.hparams.batch_size

                # assign a batch of features and labels
                batch_x = np.array(X_test[start:end])
                batch_y = np.array(y_test[start:end])
            #
            #     batch_acc, test_state = sess.run([self.accuracy, self.final_state],
            #                                   feed_dict={self.xplaceholder: np.array(batch_x),
            #                                              self.yplaceholder: np.array(batch_y)})
            #     test_acc.append(batch_acc)


                # feed testing data set into the model and tell tensorflow to run the subgraph necessary
                # to compute logit
                # pass logit value through a sigmoid activation to get prediction
                # round off to remove decimal places of predicted values
                batch_pred = tf.round(tf.nn.sigmoid(self.logits)).eval(
                    {self.xplaceholder: np.array(batch_x),
                     self.yplaceholder: np.array(batch_y)})

                test_pred.append(batch_pred)

                i += self.hparams.batch_size

            test_pred_np = np.asarray(test_pred)
            pred = test_pred_np.flatten()
            pred_size = len(pred)
            y_test = y_test[:pred_size]

            # calculate F1 score = weighted average of precision and recall
            f1 = f1_score(np.array(y_test), pred, average='macro')

            # calculate accurarcy score
            accuracy = accuracy_score(np.array(y_test), pred)

            # calculate recall = ratio of correctly predicted positive observations to all positive observations
            recall = recall_score(y_true=np.array(y_test), y_pred=pred)

            # calculate precision = ratio of correctly predicted positive observations to total predicted positive observations
            precision = precision_score(y_true=np.array(y_test), y_pred=pred)

            print("Test Accuracy: ", np.mean(test_acc))
            # print out all calculated scores
            log_results("dynamic Single LSTM Model",
                        {"F1 Score": f1, "Accuracy Score": accuracy, "Recall": recall,
                         "Precision": precision}, self.hparams.dictionary)


if __name__ == "__main__":
    print("Main function von LSTM_models.py")
