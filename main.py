"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import logging  # for logging
import time  # for measuring time

import logging_config
import util
from database_interface import Database
from ml_models import LSTM
from params import hparams, config
import talos as ta
from ml_models import lstm_model
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt

graph = []


def main():
    """Main function of project RNN-SA."""
    # initialize logging
    logging_config.init_logging()

    # create logger
    logger = logging.getLogger('RNN-SA.main.main')

    # Create a database object
    try:
        mydb = Database()
    except Exception as exc:
        logger.error('Could not create database object: {}'.format(exc))
        return

    # load data from the database
    train_X, train_y, test_X, test_y = mydb.load_data()

    # save data shape
    config['time_steps'] = train_X.shape[1]
    config['element_size'] = train_X.shape[2]

    # create dirs for checkpoints and logs
    util.create_dirs([config['checkpoint_dir'], config['tensorboard_log_dir']])

    # # ----- LSTM -----
    # # create the model
    # lstm = LSTM(hparams, config)
    #
    # # train the model
    # start_time = time.time()
    # lstm.train(train_X, train_y)
    # logger.info("Time elapsed for training: %f", time.time() - start_time)
    #
    # # evaluate the model
    # start_time = time.time()
    # loss, acc = lstm.evaluate(test_X, test_y)
    # logger.info("Test loss = %f, test accuracy = %f", loss, acc)
    # logger.info("Time elapsed for evaluation: %f", time.time() - start_time)

    # ----- Talos -----
    start_t = time.time()
    h = ta.Scan(train_X, train_y, params=hparams,
                model=lstm_model,
                dataset_name='lstm',
                experiment_no='2',
                val_split=.2,
                grid_downsample=0.01)
    print("Time elapsed: ", time.time() - start_t)


    # use filename as input
    r = ta.Reporting('lstm_2.csv')

    # heatmap correlation
    r.plot_corr(metric='val_acc', color_grades=5)

    # a four dimensional bar grid
    r.plot_bars(x='batch_size', y='val_acc', hue='hidden_layer_size', col='learning_rate')

    # show plots
    plt.show()




if __name__ == "__main__":
    main()
