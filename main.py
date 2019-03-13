"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import logging  # for logging
import time  # for measuring time

import logging_config
from database_interface import Database
from ml_models import SimpleRNN, GRU, LSTM
from util import YParams, Config


def main():
    """Main function of project RNN-SA."""
    # initialize logging
    logging_config.init_logging()

    # create logger
    logger = logging.getLogger('RNN-SA.main.main')

    # load data from the database
    mydb = Database()
    train_X, train_y, test_X, test_y = mydb.load_data()

    # ----- SimpleRNN -----
    # get hyperparameter and configuration parameter
    hparams = YParams('hparams.yaml', 'SimpleRNN')
    config = Config('config.yaml', 'SimpleRNN')

    # create the model
    simple_rnn = SimpleRNN(hparams, config)

    # train the model
    start_time = time.time()
    simple_rnn.train(train_X, train_y)
    logger.info("Time elapsed for training: %f", time.time() - start_time)

    # evaluate the model
    start_time = time.time()
    loss, acc = simple_rnn.evaluate(test_X, test_y)
    logger.info("Test loss = %f, test accuracy = %f", loss, acc)
    logger.info("Time elapsed for evaluation: %f", time.time() - start_time)

    # ----- GRU -----
    # get hyperparameter and configuration parameter
    hparams = YParams('hparams.yaml', 'GRU')
    config = Config('config.yaml', 'GRU')

    # create the model
    gru = GRU(hparams, config)

    # train the model
    start_time = time.time()
    gru.train(train_X, train_y)
    logger.info("Time elapsed for training: %f", time.time() - start_time)

    # evaluate the model
    start_time = time.time()
    loss, acc = gru.evaluate(test_X, test_y)
    logger.info("Test loss = %f, test accuracy = %f", loss, acc)
    logger.info("Time elapsed for evaluation: %f", time.time() - start_time)

    # ----- LSTM -----
    # get hyperparameter and configuration parameter
    hparams = YParams('hparams.yaml', 'LSTM')
    config = Config('config.yaml', 'LSTM')

    # create the model
    lstm = LSTM(hparams, config)

    # train the model
    start_time = time.time()
    lstm.train(train_X, train_y)
    logger.info("Time elapsed for training: %f", time.time() - start_time)

    # evaluate the model
    start_time = time.time()
    loss, acc = lstm.evaluate(test_X, test_y)
    logger.info("Test loss = %f, test accuracy = %f", loss, acc)
    logger.info("Time elapsed for evaluation: %f", time.time() - start_time)


if __name__ == "__main__":
    main()
