"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import logging  # for logging
import time  # for measuring time

import matplotlib.pyplot as plt
import talos as ta

import logging_config
import util
from database_interface import Database
from ml_models import lstm_model
from params import hparams, config, params
import csv

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
    train_X, train_y, test_X, test_y, val_X, val_y = mydb.load_data()

    # save data shape
    config['time_steps'] = train_X.shape[1]
    config['element_size'] = train_X.shape[2]

    # create dirs for checkpoints and logs
    util.create_dirs([config['checkpoint_dir'], config['tensorboard_log_dir']])

    # ----- Talos -----
    run_talos(train_X, train_y, val_X, val_y)
    do_plotting()
    #plot()


def run_talos(train_X, train_y, val_X, val_y):
    start_t = time.time()
    h = ta.Scan(
        x=train_X,
        y=train_y,
        params=hparams,
        dataset_name='lstm',
        experiment_no='3',
        model=lstm_model,
        #grid_downsample=0.1,
        x_val=val_X,
        y_val=val_y,
    )
    print("Time elapsed: ", time.time() - start_t)

def do_plotting():

    # use filename as input
    r = ta.Reporting('lstm_3.csv')

    # heatmap correlation
    r.plot_corr(metric='val_acc', color_grades=5)

    # a four dimensional bar grid
    r.plot_bars(x='num_epochs', y='val_acc', hue='batch_size', col='hidden_layer_size')

    # show plots
    plt.show()

def plot():
    x = []
    y = []

    with open('lstm_2.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)
        for row in plots:
            x.append(int(row[6]))
            y.append(float(row[2]))

    plt.plot(x, y, 'o')
    plt.plot([0, 200], [0.9275, 0.9275], 'r')
    plt.xlabel('epochs')
    plt.ylabel('val_acc')
    plt.axis([0, 200, 0.8, 1])
    plt.show()


if __name__ == "__main__":
    main()
