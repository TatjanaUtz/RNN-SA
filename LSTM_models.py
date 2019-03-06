"""Model classes."""

import logging
import math
import os
import time

import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.getcwd(), 'models\LSTM\lstm_1.ckpt')  # here the model is saved
LOG_DIR = "logs/LSTM_1"  # Here summaries of TensorBoard are saved


class LSTM_Model:
    """Single LSTM Model."""

    def __init__(self, hparams):
        """Constructor."""

        # save hyperparameters
        self.hparams = hparams

        # create placeholders for input data and labels
        self.xplaceholder = tf.placeholder(tf.float32, [None, self.hparams.time_steps,
                                                        self.hparams.element_size], name='inputs')
        self.yplaceholder = tf.placeholder(tf.float32, [None, self.hparams.num_classes],
                                           name='labels')

        # create placeholder for the sequence length
        self.seqlens = tf.placeholder(tf.int32, shape=[self.hparams.batch_size],
                                      name="sequence_lengths")

        # create LSTM model
        self.model()

        # initialize variables
        self.init = tf.global_variables_initializer()

        # create saver
        self.saver = tf.train.Saver()

        # merge summaries
        self.merged_summary = tf.summary.merge_all()

    def model(self):
        """Model function."""

        # create LSTM model with TensorFlow built-in functions
        with tf.variable_scope('lstm'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=self.hparams.hidden_layer_size,  # number of units in the LSTM cell
                use_peepholes=self.hparams.use_peepholes,
                # enable/disable diagonal/peephole connections
                cell_clip=None,
                # cell state is clipped by this value prior to the cell output activation
                initializer=None,  # initializer to use for weight and projection matrices
                num_proj=None,  # output dimensionality for the projection matrices
                proj_clip=None,
                # the projected values are clipped elementwise to within [-proj_clip, proj_clip]
                forget_bias=1.0,  # biases of the forget gate
                activation=self.hparams.activation_function,
                # activation function of the inner states
                reuse=None,  # whether to reuse variables in an existing scope
            )

            # add dropout layer if needed
            if self.hparams.keep_prob < 1.0:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                          output_keep_prob=self.hparams.keep_prob)

            # stack cells into multiple layers if needed
            self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.hparams.num_cells,
                                                    state_is_tuple=True)

            # getting an inital state of all zeros
            self.initial_state = self.cell.zero_state(self.hparams.batch_size, tf.float32)

            self.lstm_outputs, states = tf.nn.dynamic_rnn(
                cell=self.cell,  # an instance of RNNCell
                inputs=self.xplaceholder,  # the RNN inputs
                sequence_length=self.seqlens,
                # an int vector sized [batch_size], used to copy-through state
                # and zero-out outputs when past a batch element's sequence length
                initial_state=self.initial_state,  # an initial state for the RNN
                dtype=tf.float32,  # the data type for the initial state and expected output
            )

        # take only the last output of the LSTM layer
        # (Note: outputs[-1] isn't supported by tensorflow)
        lstm_outputs = tf.transpose(self.lstm_outputs, [1, 0, 2])
        last_output = tf.gather(lstm_outputs, int(lstm_outputs.get_shape()[0]) - 1,
                                name="last_lstm_output")

        # process output through a fully connected output layer
        # result = logit value of forward propagation
        self.logits = tf.contrib.layers.fully_connected(inputs=last_output,
                                                        num_outputs=self.hparams.num_classes,
                                                        activation_fn=tf.sigmoid)

        # define the loss function:
        # sigmoid_cross_entropy_with_logits because of binary classification
        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.yplaceholder
            ), name="sigmoid_cross_entropy")
            tf.summary.scalar('cross_entropy', self.cross_entropy)

        # pass loss to the optimizer: AdamOptimizer because of fairly better performance
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hparams.learning_rate  # the learning rate
            ).minimize(self.cross_entropy, name="adam_optimizer")

        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.round(self.logits), self.yplaceholder)
            self.accuracy = (tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))) * 100
            tf.summary.scalar('accuracy', self.accuracy)

    def train(self, train_dataset):
        """Train function."""
        # create logger
        logger = logging.getLogger('RNN-SA.models.Single_LSTM_Model.train')

        # calculate number of batches
        num_batches = math.ceil(train_dataset.num_samples / self.hparams.batch_size)

        # write summaries for TensorBoard to LOG_DIR
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())

        with tf.Session() as sess:
            # initialize all variables
            sess.run(self.init)

            # loop over number of iterations (epochs)
            for epoch in range(self.hparams.num_epochs):
                logger.info("Epoch %d/%d", epoch + 1, self.hparams.num_epochs)

                # save start time
                start_time = time.time()

                # create initial state
                sess.run(self.initial_state)

                # Loop over number of batches
                for step in range(int(num_batches)):
                    batch_x, batch_y, batch_seqlens = train_dataset.next_batch()

                    summary, _ = sess.run([self.merged_summary, self.optimizer],
                                          feed_dict={self.xplaceholder: batch_x,
                                                     self.yplaceholder: batch_y,
                                                     self.seqlens: batch_seqlens})

                    # add summary
                    train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        acc, loss, = sess.run([self.accuracy, self.cross_entropy],
                                              feed_dict={self.xplaceholder: batch_x,
                                                         self.yplaceholder: batch_y,
                                                         self.seqlens: batch_seqlens})
                        logger.info(
                            "Step %d, Verlust der Teilmenge= %.6f, Genauigkeit Lerndaten=%.5f",
                            step, loss, acc)

                # stop time
                stop_time = time.time()

                logger.info("Time elapsed: %f", stop_time - start_time)

            # Save model
            self.saver.save(sess, MODEL_PATH)

    def test(self, test_dataset):
        """Test function."""
        # create logger
        logger = logging.getLogger('RNN-SA.LSTM_models.test')
        test_acc = []
        test_pred = []

        with tf.Session() as sess:
            # Restore saved model
            self.saver.restore(sess, MODEL_PATH)

            # write summaries for TensorBoard to LOG_DIR
            test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())

            # calculate number of batches
            num_batches = test_dataset.num_samples // self.hparams.batch_size

            sess.run(self.cell.zero_state(test_dataset.num_samples, tf.float32))

            # Loop over number of batches
            for step in range(int(num_batches)):
                batch_x, batch_y, batch_seqlens = test_dataset.next_batch()

                summary, batch_acc = sess.run([self.merged_summary, self.accuracy],
                                              feed_dict={self.xplaceholder: batch_x,
                                                         self.yplaceholder: batch_y,
                                                         self.seqlens: batch_seqlens})
                test_writer.add_summary(summary, step)

                test_acc.append(batch_acc)

                # feed testing data set into the model and tell tensorflow to run the subgraph necessary
                # to compute logit
                # pass logit value through a sigmoid activation to get prediction
                # round off to remove decimal places of predicted values
                batch_pred = tf.round(tf.nn.sigmoid(self.logits)).eval(
                    {self.xplaceholder: batch_x,
                     self.yplaceholder: batch_y,
                     self.seqlens: batch_seqlens})

                test_pred.append(batch_pred)

            logger.info("Genauigkeit Test: %f", np.mean(test_acc))

            test_pred_np = np.asarray(test_pred)
            pred = test_pred_np.flatten()
            y_test = test_dataset.labels
            #
            # # calculate F1 score = weighted average of precision and recall
            # f1 = f1_score(np.array(y_test), pred, average='macro')
            #
            # # calculate accurarcy score
            # accuracy = accuracy_score(np.array(y_test), pred)
            #
            # # calculate recall = ratio of correctly predicted positive observations to all positive observations
            # recall = recall_score(y_true=np.array(y_test), y_pred=pred)
            #
            # # calculate precision = ratio of correctly predicted positive observations to total predicted positive observations
            # precision = precision_score(y_true=np.array(y_test), y_pred=pred)
            #
            # print("Test Accuracy: ", np.mean(test_acc))
            # # print out all calculated scores
            # log_results("dynamic Single LSTM Model",
            #             {"F1 Score": f1, "Accuracy Score": accuracy, "Recall": recall,
            #              "Precision": precision}, self.hparams.dictionary)


if __name__ == "__main__":
    print("Main function von LSTM_models.py")
