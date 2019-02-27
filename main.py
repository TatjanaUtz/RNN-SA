"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""
import time

from sklearn.model_selection import train_test_split

import logging_config
import LSTM_models
from database import Database
from utils import YParams


def main():
    """Main function."""
    # initialize logging
    logging_config.init_logging()

    # get the dataset from the database
    my_database = Database()
    features, labels = my_database.read_all_tasksets_2D()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                        shuffle=False, random_state=42)

    lstm_model = LSTM_models.LSTM_Model()
    lstm_model.train(X_train, y_train)
    lstm_model.test(X_test, y_test)


if __name__ == "__main__":
    # main()
    # print("Main function of RNN-SA/main.py")

    # initialize logging
    logging_config.init_logging()

    my_db = Database()

    hparams = YParams('hparams.yaml', 'LSTM')

    start_time = time.time()
    features, labels = my_db.read_all_tasksets()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=hparams.test_size,
                                                        shuffle=False, random_state=42)
    print("Time elapsed for data preprocessing: ", time.time() - start_time)

    lstm_model = LSTM_models.LSTM_Model(hparams)

    # start_time = time.time()
    # lstm_model.train(X_train, y_train)
    # print("Time elapsed for training: ", time.time() - start_time)

    start_time = time.time()
    lstm_model.test(X_test, y_test)
    print("Time elapsed for test: ", time.time() - start_time)
