"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""
from sklearn.model_selection import train_test_split

import logging_config
from database import Database
import models


def main():
    """Main function."""
    # initialize logging
    logging_config.init_logging()

    # get the dataset from the database
    my_database = Database()
    features, labels = my_database.read_all_tasksets_2D()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                        shuffle=False, random_state=42)

    lstm_model = models.Single_LSTM_Model()
    lstm_model.train(X_train, y_train)
    lstm_model.test(X_test, y_test)


if __name__ == "__main__":
    # main()
    # print("Main function of RNN-SA/main.py")

    my_db = Database()
    features_3D, labels_3D = my_db.read_all_tasksets_3D()

    X_train, X_test, y_train, y_test = train_test_split(features_3D, labels_3D, test_size=0.2,
                                                        shuffle=False, random_state=42)

    lstm_model = models.dynamic_Single_LSTM_Model()
    lstm_model.train(X_train, y_train)
    lstm_model.test(X_test, y_test)


    print("Dummy for debugging")
