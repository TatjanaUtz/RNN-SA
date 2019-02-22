"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""
from random import shuffle

import numpy as np
import tensorflow as tf

from database import Database
from dataset import Dataset
from logging_config import init_logging


def main_tutorial():
    """Main function for tutorial."""
    # generate input data = tasksets
    train_input = ['{0:020b}'.format(i) for i in range(2 ** 20)]
    shuffle(train_input)
    train_input = [map(int, i) for i in train_input]
    ti = []
    for i in train_input:
        temp_list = []
        for j in i:
            temp_list.append([j])
        ti.append(np.array(temp_list))
    train_input = ti

    # generate output data = labels
    train_output = []
    for i in train_input:
        count = 0
        for j in i:
            if j[0] == 1:
                count += 1
        temp_list = ([0] * 21)
        temp_list[count] = 1
        train_output.append(temp_list)

    # split datasets
    NUM_EXAMPLES = 10000
    test_input = train_input[NUM_EXAMPLES:]
    test_output = train_input[NUM_EXAMPLES:]
    train_input = train_input[:NUM_EXAMPLES]
    train_output = train_output[:NUM_EXAMPLES]

    # variables for input data and target data
    # dimension = [batch_size, sequence length, input dimension], None = unknown (determined at runtime)
    data = tf.placeholder(tf.float32, [None, 20, 1])
    target = tf.placeholder(tf.float32, [None, 21])  # dimension = [batch_size, output_size]

    # create LSTM cell
    num_hidden = 24
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

    # unroll the network, calculate output in val
    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    val = tf.transpose(val,
                       [1, 0, 2])  # transpose the output to switch batch size with sequence size
    last = tf.gather(val, int(val.get_shape()[0]) - 1)  # take only last output of every sequence

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    # final transformation of the outputs of the LSTM and map it to the output classes
    prediction = tf.nn.softmax(
        tf.matmul(last, weight) + bias)  # dimension = [batch_size, output_size]
    # calculate cross entropy loss = cost function
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    # prepare optimization function
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    # calculate error on test data
    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # execution of graph
    init_op = tf.initialize_all_variables()  # initialize all variables
    sess = tf.Session()  # create session
    sess.run(init_op)

    # begin training process
    batch_size = 1000  # determine batch size
    no_of_batches = int(len(train_input) / batch_size)  # calculate number of batches
    epoch = 5000
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):  # iterate over all batches
            # get input and output data
            inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
            ptr += batch_size
            # run minimize (optimizer function) to minimize cost
            sess.run(minimize, {data: inp, target: out})
        print("Epoch - ", str(i))
    incorrect = sess.run(error, {data: test_input, target: test_output})
    print(sess.run(prediction, {data: [
        [[1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1],
         [1], [0]]]}))
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()


def main_RNN_for_SA():
    """Main function of project.

    Run this method to perform schedulability analysis wit recurrent neural network (RNN).
    """
    logger = init_logging()  # initialize logging

    # Import data
    my_database = Database()
    tasksets, labels = my_database.read_all_tasksets()
    my_dataset = Dataset(tasksets, labels)

    train_input = my_dataset.train_tasksets
    train_output = my_dataset.train_labels
    test_input = my_dataset.test_tasksets
    test_output = my_dataset.test_labels

    # define dimensions
    batch_size = 32  # how many samples at once?
    sequence_length = 4  # size of task-set (1 - 4 tasks)
    input_dimension = 9  # number of attributes per task
    output_size = 1  # size of the output (0 or 1)
    num_hidden = 32  # number of hidden units in the LSTM cell
    epoch = 50  # number of epochs
    learning_rate = 0.0001

    # placeholder for input and label data
    input_placeholder = tf.placeholder(
        dtype=tf.float32,  # type of elements
        shape=[batch_size, sequence_length * input_dimension],  # shape of tensor (None=any shape)
        name='input')  # name for operation (optional)
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_size],
                                       name='label')

    # create a LSTM cell
    cell = tf.nn.rnn_cell.LSTMCell(
        num_units=num_hidden,  # number of units in the LSTM cell
        use_peepholes=False,  # en-/disable diagonal/peephole connections
        cell_clip=None,
        # cell state is clipped by this value prior to cell output activation (optional)
        initializer=None,  # initializer to use for weight and projection matrices (optional)
        num_proj=None,
        # output dimensionality for projection matrices (None=no projection performed) (optional)
        proj_clip=None,
        # projection values clipped elementwise to within [-proj_clip, proj_clip] (optional)
        forget_bias=1.0,  # biases of forget gate
        state_is_tuple=True,  # returned states are 2-tuples of c_state and m_state
        activation=None,  # activation function of inner states (default=tanh)
        reuse=None,  # reuse variables in existing scope (optional)
        name=None,  # name of the layer
        dtype=None)  # default dtype of layer

    # TODO: DropoutWrapper needed?
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=cell,  # RNNCell, projection to output_size is added to it
        input_keep_prob=1.0,  # input keep probability (1=no input dropout)
        output_keep_prob=0.5,  # output keep probability (1=no output dropout)
        state_keep_prob=1.0,  # output keep probability (1=no output dropout)
        variational_recurrent=False,
        # True=same dropout pattern is applied across all time steps per run call
        input_size=None,
        # depth(s) of input tensors expected to be passed into the DropoutWrapper (optional)
        dtype=None,  # dtype of input, state and output tensors (optional)
        seed=None,  # randomness seed (optional)
        dropout_state_filter_visitor=None)  # TODO: what is this?

    # TODO: define initial state of the network - needed?
    initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    rnn_outputs, rnn_states = cell(input_placeholder, initial_state)

    # TODO: what is this doing??
    logit = tf.layers.dense(rnn_outputs, 1)
    prediction = tf.sigmoid(logit)

    # calculate cross entropy loss = cost function
    loss = tf.reduce_mean(tf.losses.log_loss(
        labels=label_placeholder,  # ground truth output tensor, same dimension as 'predictions'
        predictions=prediction,  # predicted outputs
        weights=1.0,  # optional
        epsilon=1e-07,  # small increment to add to avoid taking a log of zero
        scope=None,  # scope for operations perfomred in computing loss
        loss_collection=tf.GraphKeys.LOSSES,  # collection to which loss will be added
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS))  # type of reduction to apply to loss

    # prepare optimization function
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimize = optimizer.minimize(loss)

    # execution of graph
    init_op = tf.global_variables_initializer()  # initialize all variables
    sess = tf.Session()  # create session
    sess.run(init_op)

    # begin training process
    no_of_batches = int(len(train_input) / batch_size)  # calculate number of batches
    for i in range(epoch):
        # train loss
        total_loss = 0
        ptr = 0
        for j in range(no_of_batches):  # iterate over all batches
            # get input and output data
            inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
            ptr += batch_size

            # run minimize (optimizer function) to minimize cost
            _, loss_value, outputs = sess.run([minimize, loss, prediction],
                                              {input_placeholder: inp, label_placeholder: out})
            total_loss += loss_value
            print('loss value', loss_value, ' ', j)
        print("Epoch - ", str(i))

    ptr = 0
    for step in range(no_of_batches):
        inp, out = test_input[ptr:ptr + batch_size], test_output[prt:prt + batch_size]
        ptr += batch_size
        _, loss_value, outputs = sess.run([minimize, loss, prediction],
                                          {input_placeholder: inp, label_placeholder: out})
        total_loss += loss_value
    print("Validation loss: ", total_loss)

    # incorrect = sess.run(error, {input_placeholder: test_input, label_placeholder: test_output})
    # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()


def main():
    """sceleton for main function."""
    # define dimensions of input data
    sequence_length = 4
    input_size = 9
    output_size = 1

    # read data
    # TODO: import data as tensorflow dataset
    my_database = Database()
    input_data, labels = my_database.read_all_tasksets()
    my_dataset = Dataset(input_data, labels)

    # create placeholders for input data and labels
    input_placeholder = tf.placeholder(
        dtype=tf.float32,  # type of elements
        # shape of the tensor (optional), None = tensor of any shape
        shape=[None, sequence_length * input_size],
        name='input')  # name for the operation (optional)
    label_placeholder = tf.placeholder(tf.float32, [None, output_size])

    ##### DEFINE MODEL #####
    # Layer settings
    num_hidden_nodes = 200
    num_output_nodes = 1
    num_units = 200
    num_layers = 3
    dropout = tf.placeholder(tf.float32)

    # Create RNN-cells
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.GRUCell(num_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRnnCell(cells)

    # batch size x time steps x features
    data = tf.placehol


    # Define weights
    w0 = tf.Variable(tf.random_normal([sequence_length * input_size, num_hidden_nodes]))
    w1 = tf.Variable(tf.random_normal([num_hidden_nodes, num_hidden_nodes]))
    w2 = tf.Variable(tf.random_normal([num_hidden_nodes, num_hidden_nodes]))
    w3 = tf.Variable(tf.random_normal([num_hidden_nodes, num_output_nodes]))

    # Define biases
    b0 = tf.Variable(tf.random_normal([num_hidden_nodes]))
    b1 = tf.Variable(tf.random_normal([num_hidden_nodes]))
    b2 = tf.Variable(tf.random_normal([num_hidden_nodes]))
    b3 = tf.Variable(tf.random_normal([num_hidden_nodes]))

    # Create layers
    layer_1 = tf.add(tf.matmul(input_placeholder, w0), b0)
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, w1), b1)
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, w2), b2)
    layer_3 = tf.nn.relu(layer_3)
    out_layer = tf.matmul(layer_3, w3) + b3

    # Compute loss
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=label_placeholder))
    loss = -1. *  tf.reduce_sum(label_placeholder * tf.log(out_layer) + (1. - label_placeholder) * (1. - tf.log(out_layer)))

    # Create optimizer
    learning_rate = 0.005
    num_epochs = 350
    batch_size = 50
    num_batches = int(my_dataset.train.num_samples / batch_size)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Launch session
    with tf.Session() as sess:
        sess.run(init)

        # Loop over epochs
        for epoch in range(num_epochs):

            # Loop over batches
            for batch in range(num_batches):
                input_batch, label_batch = my_dataset.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={input_placeholder: input_batch, label_placeholder: label_batch})

        # Determine success rate
        prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(label_placeholder, 1))
        success = tf.reduce_mean(tf.cast(prediction, tf.float32))
        print('Success rate: ', sess.run(success, feed_dict={input_placeholder: my_dataset.test.input, label_placeholder: my_dataset.test.labels}))


if __name__ == "__main__":
    # my_database = Database()
    # tasksets, labels = my_database.read_all_tasksets()
    # my_dataset = Dataset(tasksets, labels)


    main()


