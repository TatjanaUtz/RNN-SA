"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""
import time


import logging_config
import LSTM_models
from dataset import Dataset
from util import YParams
from database_interface import Database


def main():
    """Main function."""
    # initialize logging
    logging_config.init_logging()

    # get hyperparameters
    hparams = YParams('hparams.yaml', 'LSTM')

    # get the dataset from the database
    my_dataset = Dataset(hparams=hparams)

    # create LSTM model
    lstm_model = LSTM_models.LSTM_Model(hparams)

    start_time = time.time()
    lstm_model.train(my_dataset.train)
    print("Time elapsed for training: ", time.time() - start_time)

    start_time = time.time()
    lstm_model.test(my_dataset.test)
    print("Time elapsed for test: ", time.time() - start_time)



if __name__ == "__main__":
    main()
    # print("Main function of RNN-SA/main.py")

