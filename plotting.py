"""File for different plots for visualisation of the results of hyperparameter optimization.

- Single Line Plot: plot the validation accuracy as a function of one hyperparamter
- Confusion Matrix: plot the confusion matrix of test dataset
- Correlation Matrix: plot correlation matrix between validation accuracy and all hyperparamters
"""
import csv
import os
import time

import keras
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import talos

import main
from params import config


########################
### Single Line Plot ###
########################
def plot_num_epochs():
    """Plot validation accuracy as function of num_epochs."""
    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    filepath = os.path.join(os.getcwd(), "experiments", "LSTM", "LSTM_num_epochs.csv")
    with open(filepath, 'r') as csvfile:  # open csv file
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)  # read header
        for row in plots:  # iterate over all rows and read data
            x.append(int(row[7]))  # hyperparameter to plot
            y.append(float(row[2]))  # validation accuracy

    plt.plot(x, y, 'o')  # line plot of y = f(x)
    # plt.plot([128, 128], [0, 1], 'r--')  # vertical line
    plt.plot([0, 200], [0.932, 0.932], 'r')  # horizontal line

    plt.xlabel('num_epochs')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 200, 0.8, 1])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]
    # plt.xticks([0, 32, 64, 128, 200, 256, 400, 512, 600, 800, 1024])  # ticks of x-axis
    # plt.yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])  # ticks of y-axis

    plt.show()  # show all plots


def plot_batch_size():
    """Plot validation accuracy as function of batch_size."""
    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    filepath = os.path.join(os.getcwd(), "experiments", "LSTM", "LSTM_batch_size.csv")
    with open(filepath, 'r') as csvfile:  # open csv file
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)  # read header
        for row in plots:  # iterate over all rows and read data
            x.append(int(row[6]))  # hyperparameter to plot
            y.append(float(row[2]))  # validation accuracy

    plt.plot(x, y, 'o')  # line plot of y = f(x)
    plt.plot([128, 128], [0, 1], 'r--')  # vertical line
    plt.plot([0, 1050], [0.9574916288153964, 0.9574916288153964], 'r')  # horizontal line

    plt.xlabel('batch_size')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 1050, 0.9, 0.96])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]
    plt.xticks([0, 32, 64, 128, 200, 256, 400, 512, 600, 800, 1024])  # ticks of x-axis
    plt.yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])  # ticks of y-axis

    plt.show()  # show all plots


def plot_hidden_layer_size():
    """Plot validation accuracy as function of hidden_layer_size."""
    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    filepath = os.path.join(os.getcwd(), "experiments", "LSTM", "LSTM_hidden_layer_size.csv")
    with open(filepath, 'r') as csvfile:  # open csv file
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)  # read header
        for row in plots:  # iterate over all rows and read data
            x.append(int(row[10]))  # hyperparameter to plot
            y.append(float(row[2]))  # validation accuracy

    plt.figure(1, (12.8, 4.8))  # create figure with specific size

    ### subplot for all data points ###
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')  # line plot of y = f(x)
    plt.plot([0, 5000], [0.977, 0.977], 'r')  # horizontal line

    plt.xlabel('hidden_layer_size')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 5000, 0.88, 0.98])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]

    ### subplot for interesting data points ###
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'o')  # line plot of y = f(x)
    plt.plot([319, 319], [0, 1], 'r--')  # vertical line
    plt.plot([0, 1000], [0.977, 0.977], 'r')  # horizontal line

    plt.xlabel('hidden_layer_size')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 1000, 0.88, 0.98])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])  # ticks of x-axis
    # plt.yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])  # ticks of y-axis

    plt.show()  # show all plots


def plot_num_cells():
    """Plot validation accuracy as function of num_cells."""
    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    filepath = os.path.join(os.getcwd(), "experiments", "LSTM", "LSTM_num_cells.csv")
    with open(filepath, 'r') as csvfile:  # open csv file
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)  # read header
        for row in plots:  # iterate over all rows and read data
            x.append(int(row[9]))  # hyperparameter to plot
            y.append(float(row[2]))  # validation accuracy

    plt.plot(x, y, 'o')  # line plot of y = f(x)
    # plt.plot([128, 128], [0, 1], 'r--')  # vertical line
    plt.plot([0, 11], [0.9847, 0.9847], 'r')  # horizontal line

    plt.xlabel('num_cells')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 11, 0.95, 1])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # ticks of x-axis
    # plt.yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])  # ticks of y-axis

    plt.show()  # show all plots


def plot_keep_prob():
    """Plot validation accuracy as function of num_cells."""
    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    filepath = os.path.join(os.getcwd(), "experiments", "LSTM", "LSTM_keep_prob.csv")
    with open(filepath, 'r') as csvfile:  # open csv file
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)  # read header
        for row in plots:  # iterate over all rows and read data
            x.append(float(row[8]))  # hyperparameter to plot
            y.append(float(row[2]))  # validation accuracy

    plt.plot(x, y, 'o')  # line plot of y = f(x)
    # plt.plot([128, 128], [0, 1], 'r--')  # vertical line
    plt.plot([0, 1], [0.9843, 0.9843], 'r')  # horizontal line

    plt.xlabel('keep_prob')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 1, 0.95, 1])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # ticks of x-axis
    # plt.yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])  # ticks of y-axis

    plt.show()  # show all plots


########################
### Confusion Matrix ###
########################
def get_confusion_matrix():
    """Get and plot confusion matrix (tp, fp, tn, fn) of a ML model."""
    # load weights
    print("Loading model...")
    model = keras.models.load_model(os.path.join(config['checkpoint_dir'], "weights.best.hdf5"))
    print("Model successfully loaded!")

    # load data
    print("Loading data...")
    data = main.load_data(os.getcwd(), "panda_v3.db")
    print("Data successfully loaded!")

    # stack all datasets together
    dataset_X = np.concatenate((data['train_X'], data['val_X'], data['test_X']), axis=0)
    dataset_y = np.concatenate((data['train_y'], data['val_y'], data['test_y']), axis=0)

    # get predictions
    print("Predicting classes...")
    start_t = time.time()
    y_pred = model.predict_classes(dataset_X)
    end_t = time.time()
    print("Classes successfully predicted!")
    print("Time elapsed: %f s \n" % (end_t - start_t))

    # get and print confusion matrix
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(dataset_y, y_pred).ravel()
    print("            | task-set schedulable | task-set not schedulable")
    print("-------------------------------------------------------------")
    print("SA positive | tp = %d              | fp = %d" % (tp, fp))
    print("-------------------------------------------------------------")
    print("SA negative | fn = %d              | tn = %d" % (fn, tn))


##########################
### Correlation Matrix ###
##########################
def get_correlation_matrix():
    ### CORRELATION ###
    # plot correlation between all hyperparameters and the validation accuracy with Talos
    filepath = os.path.join(os.getcwd(), "experiments", "LSTM", "LSTM_all_experiments.csv")
    r = talos.Reporting(filepath)  # open and read csv file

    r.plot_corr()  # plot correlation matrix

    plt.show()  # show all plots


if __name__ == "__main__":
    # plot_num_epochs()
    # plot_batch_size()
    # plot_hidden_layer_size()
    # plot_num_cells()
    # plot_keep_prob()

    get_confusion_matrix()

    # get_correlation_matrix()
