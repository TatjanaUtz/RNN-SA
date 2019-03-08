"""Model classes."""

import logging
import math
import os
import time

import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.getcwd(), 'models\LSTM\lstm_1.ckpt')  # here the model is saved
LOG_DIR = "logs/LSTM_1"  # Here summaries of TensorBoard are saved


class Base_Model:
    """Base model for different RNN models."""

    def __init__(self, hparams):
        """Constructor."""
        self.hparams = hparams  # save hyperparameters (s. hparams.yaml)

        ### construct placeholders ###
        # input data of shape [batch_size X time_steps X element_size]
        self.xplaceholder = tf.placeholder(
            dtype=tf.float32,  # the type of elements in the tensor to be fed
            # the shape of the tensor to be fed (optional, default: None)
            shape=(self.hparams.batch_size, self.hparams.time_steps, self.hparams.element_size),
            name='input')  # a name for the operation (optional, default: None)

        # labels of shape [batch_size X num_classes]
        self.yplaceholder = tf.placeholder(
            dtype=tf.float32,
            shape=(self.hparams.batch_size, self.hparams.num_classes),
            name='labels')

        # sequence lengths of shape [batch_size] (requirement of TensorFlow)
        self.seqlens = tf.placeholder(dtype=tf.int32, shape=(self.hparams.batch_size),
                                      name='sequence_lengths')


class LSTM_Model(Base_Model):
    """Single LSTM Model."""

    def __init__(self, hparams):
        """Constructor."""

        # call __init__ method of super class
        super().__init__(hparams=hparams)

        # initialize variables
        self.init = tf.global_variables_initializer()


        # create LSTM model
        self.initial_state, self.logits = self.model()

        # calculate loss
        self.loss = self.get_loss()

        # get optimizer
        self.optimizer = self.get_optimizer()

        # calculate accuracy
        self.accuracy = self.get_accuracy()


        # create saver
        self.saver = tf.train.Saver()

        # merge summaries
        self.merged_summary = tf.summary.merge_all()

    def model(self):
        """Model function."""

        # create LSTM model with TensorFlow built-in functions
        with tf.variable_scope('lstm'):
            # create LSTM cell
            lstm_cell = tf.nn.rnn_cell.LSTMCell(
                # int, the number of units in the LSTM cell
                num_units=self.hparams.hidden_layer_size,
                # bool, set True to enable diagonal/peephole connections (default: False)
                use_peepholes=False,
                # a float value, if provided the cell state is clipped by this value
                # prior to the cell output activation (optional, default: None)
                cell_clip=None,
                # the initializer to use for the weight and projection matrices
                # (optional, default: None)
                initializer=None,
                # int, the output dimensionality for the projection matrices, if None no projection
                # is performed (optional, default: None)
                num_proj=None,
                # a float value, if num_proj > 0 and proj_clip is provided, then the projected
                # values are clipped elementwise to within [-proj_clip, proj_clip]
                # (optional, default: None)
                proj_clip=None,
                # deprecated, will be removed by Jan. 2017, use a variable_scope partitioner instead
                # (default: None)
                num_unit_shards=None,
                # deprecated, will be removed by Jan. 2017, use a variable_scope partitioner instead
                # (default: None)
                num_proj_shards=None,
                # biases of the forget gate are initialized by default to 1 in order to reduce the
                # scale of forgetting at the beginning of the training, must set it manually to 0.0
                # when restoring from CudnnLSTM trained checkpoints (default: 1.0)
                forget_bias=1.0,
                #  if True accepted and returned states are 2-tuples of the c_state and m_state
                #  if False they are concatenated along the column axis, this latter behavior will
                #  soon be deprecated (default: True)
                state_is_tuple=True,
                # activation function of the inner states, it could be a string that is within
                # Keras activation function names (default: 'tanh')
                activation=self.hparams.activation_function,
                # Python boolean describing whether to reuse variables in an existing scope, if not
                # True and the existing scope already has the given variables an error is raised
                # (optional, default: None)
                reuse=None,
                # string, the name of the layer, layers with the same name will share weights, but
                # to avoid mistakes we require reuse=True in such cases (default: None)
                name=None,
                # default dtype of the layer (default of None means use the type of the first input)
                # required when build is called before call (default: None)
                dtype=None
            )

            # add dropout layer if needed
            if self.hparams.keep_prob < 1.0:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=lstm_cell,  # an RNNCell, a projection to output_size is added to it
                    # unit Tensor or float between 0 and 1, output keep probability; if it is
                    # constant and 1, no output dropout will be added (default: 1.0)
                    output_keep_prob=self.hparams.keep_prob,
                )

            # stack cells into multiple layers if needed
            cell = tf.nn.rnn_cell.MultiRNNCell(
                # list of RNNCells that will be composed in this order
                cells=[lstm_cell] * self.hparams.num_cells,
                # if True, accepted and returned states are n-tuples, where n = len(cells),
                # if False, the states are all concatenated along the column axis, this latter
                # behavior will soon be deprecated (default: True)
                state_is_tuple=True
            )

            # getting an inital state of all zeros
            initial_state = cell.zero_state(
                # int, float, or unit Tensor representing the batch size
                batch_size=self.hparams.batch_size,
                dtype=tf.float32  # the data type to use for the state
            )

            # run LSTM network
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                cell=cell,  # an instance of RNNCell
                # the RNN inputs; if time_major == False this must be a Tensor of shape
                # [batch_size, max_time, ...], or a nested tuple of such elements;
                # if time_major == True this must be a Tensor of shape [max_time, batch_size, ...],
                # or a nested tuple of such elements
                inputs=self.xplaceholder,
                # an int32/int64 vector sized [batch_size], used to copy-through state and zero-out
                # outputs when past a batch element's sequence length, so it's more for performance
                # than correctness (optional, default: None)
                sequence_length=self.seqlens,
                # an initial state for the RNN, if cell.state_size is an integer, this must be a
                # Tensor of appropriate type and shape [batch_size, cell.state_size], if
                # cell.state_size is a tuple, this should be a tuple of tensors having shapes
                # [batch_size, s] for s in cell.state_size (optional, default: None)
                initial_state=initial_state,
                # the data type for the initial state and expected output, required if initial_state
                # is not provided or RNN state has a heterogeneous dtype (optional, default: None)
                dtype=tf.float32,
            )

        # take only the last output of the LSTM network
        # (Note: outputs[-1] isn't supported by tensorflow)
        lstm_outputs = tf.transpose(
            a=lstm_outputs,  # a tensor
            perm=[1, 0, 2],  # a permutation of the dimension of a
            name='transpose_lstm_outputs',  # a name for the operation (optional)
        )
        last_output = tf.gather(
            # a tensor, the tensor from which to gather values, must be at least rank axis + 1
            params=lstm_outputs,
            # a Tensor, must be one of the following types: int32, int64; index tensor, must be in
            # range [0, params.shape[axis])
            indices=int(lstm_outputs.get_shape()[0]) - 1,
            name="last_lstm_output",  # a name for the operation (optional, default: None)
            # a Tensor, must be one of the following types: int32, int64; the axis in params to
            # gather indices from, defaults to the first dimension (0), supports negative indexes
            axis=0
        )

        with tf.variable_scope('output_layer'):
            # process output through a fully connected output layer
            # result = logit value of forward propagation
            logits = tf.contrib.layers.fully_connected(
                # a tensor of at least rank 2 and static value for the last dimension
                inputs=last_output,
                num_outputs=self.hparams.num_classes,
                # integer or long, the number of output units in the layer
                # activation function, explicitly set it to None to skip it and maintain a linear
                # activation (default: tf.nn.relu)
                activation_fn=tf.sigmoid)
            # self.logits = tf.layers.dense(last_output, self.hparams.num_classes, name='logits')

        return initial_state, logits

    def get_loss(self):
        # define the loss function:
        # sigmoid_cross_entropy_with_logits because of binary classification
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                # the tensor to reduce, should have numeric type
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.yplaceholder,  # a Tensor of the same type and shape as logits
                    logits=self.logits,  # a Tensor of type float32 or float64
                    # a name for the operation (optional, default: None)
                    name='sigmoid_cross_entropy'),
                # a name for the operation (optional, default: None)
                name="reduce_mean"
            )
            # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     logits=self.logits,
            #     labels=self.yplaceholder),
            #     name='softmax_cross_entropy')
            tf.summary.scalar('loss', loss)

        return loss

    def get_optimizer(self):
        # pass loss to the optimizer: AdamOptimizer because of fairly better performance
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                # a Tensor or a floating point value, the learning rate
                learning_rate=self.hparams.learning_rate,
                # name for the operations created when applying gradients (optional, default: 'Adam')
                name='Adam'
            ).minimize(
                loss=self.loss,  # a Tensor containing the value to minimize
                name='adam_optimizer'  # name for the returned operation (optional, default: None)
            )

        return optimizer

    def get_accuracy(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                x=tf.round(self.logits),    # a tensor
                y=self.yplaceholder, # a tensor, must have the same type as x
                name='correct_prediction') # a name for the operation (optional, default: None)
            # self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.yplaceholder, 1))

            accuracy = tf.reduce_mean(
                # the tensor to reduce, should have numeric type
                input_tensor=tf.cast(
                    # a Tensor or SparseTensor or IndexedSlices of numeric type
                    x=correct_prediction,
                    # the destination type, the list of supported dtypes is the same as x
                    dtype=tf.float32,
                    # a name for the operation (optional, default: None)
                    name='cast_correct_prediction'),
                name='accuracy' # a name for the operation (optional, default: None)
            )
            tf.summary.scalar('accuracy', accuracy)

        return accuracy

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

                    # feed dictionary
                    feed = {self.xplaceholder: batch_x,
                            self.yplaceholder: batch_y,
                            self.seqlens: batch_seqlens}

                    summary, _ = sess.run([self.merged_summary, self.optimizer],
                                          feed_dict=feed)

                    # add summary
                    train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        acc, loss, = sess.run([self.accuracy, self.loss],
                                              feed_dict=feed)
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

            sess.run(self.initial_state)

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

# lstm_cell = tf.keras.layers.LSTMCell(
#     # postive integer, dimensionality of the output space
#     units=self.hparams.hidden_layer_size,
#     # activation function to use, default: 'tanh', None: no activation is applied
#     activation=self.hparams.activation_function,
#     # activation function to use for the recurrent step, default: 'hard_sigmoid',
#     # None: no activation is applied
#     recurrent_activation='hard_sigmoid',
#     use_bias=True,  # boolean, whether the layer uses a bias vector (default: True)
#     # initializer for the kernel weights matrix, used for the linear transformation of
#     # the inputs (default: 'glorot_uniform')
#     kernel_initializer='glorot_uniform',
#     # initializer for the recurrent_kernel weights matrix, used for the linear
#     # transformation of the recurrent state (default: 'orthogonal)
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',  # initializer for the bias vector (default: 'zeros')
#     # boolean, if True, add 1 to the bias of the forget gate at initialization, setting
#     # it to true will also force bias_initializer="zeros", this is recommended in
#     # Jozefowicz et al. (default: True)
#     unit_forget_bias=True,
#     # regularizer function applied to the kernel weights matrix (default: None)
#     kernel_regularizer=None,
#     # regularizer function applied to the recurrent_kernel weights matrix (default:None)
#     recurrent_regularizer=None,
#     # regularizer function applied to the bias vector (default: None)
#     bias_regularizer=None,
#     # constraint function applied to the kernel weights matrix (default: None)
#     kernel_constraint=None,
#     # constraint function applied to the recurrent_kernel weights matrix (default: None)
#     recurrent_constraint=None,
#     # constraint function applied to the bias vector (default: None)
#     bias_constraint=None,
#     # float between 0 and 1, fraction of the units to drop for the linear transformation
#     # of the inputs (default: 0.0)
#     dropout=0.0,
#     # float between 0 and 1, fraction of the units to drop for the linear transformation
#     # of the recurrent state (default: 0.0)
#     recurrent_dropout=0.0,
#     # implementation mode, either 1 or 2, mode 1 will structure its operations as a
#     # larger number of smaller dot products and additions, whereas mode 2 will batch
#     # them into fewer, larger operations, these modes will have different performance
#     # profiles on different hardware and for different applications (default: 1)
#     implementation=1
# )

# self.cell = tf.keras.layers.StackedRNNCells(
#     cells=[lstm_cell] * self.hparams.num_cells  # list of RNN cell instances
# )

# self.lstm_outputs, state = keras.layers.RNN(
#     cell=self.cell,  # a RNN cell instance or a list of RNN cell instances
#     # boolean, whether to return the last output in the output sequence or the full
#     # sequence (default: False)
#     return_sequences=False,
#     # boolean, whether to return the last state in addition to the output
#     # (default: False)
#     return_state=False,
#     # boolean, if True process the input sequence backward and return the revrsed
#     # sequence (default: False)
#     go_backwards=False,
#     # boolean, if True the last state for each sample at index i in a batch will be used
#     # as initial state for the sample of index i in the following batch (default: False)
#     stateful=False,
#     # boolean, if True the network will be unrolled else a symbolic loop will be used,
#     # unrolling can speed-up a RNN, although it tends to be more memory-intensive,
#     # unrolling is only suitable for short sequences (default: False)
#     unroll=False,
#     # the shape format of the inputs and outputs tensors, if True the inputs and outputs
#     # will be in shape (timesteps, batch, ...), whereas in the False case it will be
#     # (batch, timesteps, ...) (default: False)
#     time_major=False
# )
