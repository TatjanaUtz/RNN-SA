"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import time

from tensorflow import keras

import logging_config
from database_interface import Database
from ml_models import LSTMModel
from util import YParams, Config


def main():
    """Main function of project RNN-SA."""
    # initialize logging
    logging_config.init_logging()

    # load data from the database
    mydb = Database()
    train_X, train_y, test_X, test_y = mydb.load_data()

    # prepare the data: pad sequences to uniform length
    train_X = keras.preprocessing.sequence.pad_sequences(train_X, padding='post', dtype=list)
    test_X = keras.preprocessing.sequence.pad_sequences(test_X, padding='post', dtype=list)

    # get hyperparameter and configuration parameter for LSTM
    hparams_LSTM = YParams('hparams.yaml', 'LSTM')
    config_LSTM = Config('config.yaml', 'LSTM')

    # create a LSTM model
    LSTM = LSTMModel(hparams_LSTM, config_LSTM)

    # train the LSTM model
    start_time = time.time()
    LSTM.train(train_X, train_y)
    print("Time elapsed for training: ", time.time() - start_time)

    # evaluate the LSTM model
    start_time = time.time()
    LSTM.evaluate(test_X, test_y)
    print("Time elapsed for evaluation: ", time.time() - start_time)


if __name__ == "__main__":
    main()
