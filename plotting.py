"""File for different plots for visualisation of the results of hyperparameter optimization.

- Single Line Plot: plot the validation accuracy as a function of one hyperparamter
- Correlation: plot correlation matrix between validation accuracy and all hyperparamters
"""


def single_line_plot():
    ### SINGLE LINE PLOT ###
    import csv

    import matplotlib.pyplot as plt

    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    with open('LSTM_num_cells.csv', 'r') as csvfile:  # open csv file
        plots = csv.reader(csvfile, delimiter=',')
        header = next(plots)  # read header
        for row in plots:  # iterate over all rows and read data
            x.append(int(row[9]))  # hyperparameter to plot
            y.append(float(row[2]))  # validation accuracy

    plt.plot(x, y, 'o')  # line plot of y = f(x)
    # plt.plot([200, 200], [0, 1], 'r--')  # vertical line
    # plt.plot([0, 5000], [0.977, 0.977], 'r')  # horizontal line

    plt.xlabel('hidden_layer_size')  # label of x-axis
    plt.ylabel('val_acc')  # label of y-axis
    plt.axis([0, 1000, 0.88, 0.98])  # limits of x- and y-axis: [min_x, max_x, min_y, max_y]
    # plt.xticks([0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])    # ticks of x-axis
    # plt.yticks([0.9, 0.92, 0.94, 0.96, 0.98, 1])  # ticks of y-axis

    plt.show()  # show all plots


def correlation():
    ### CORRELATION ###
    # plot correlation between all hyperparameters and the validation accuracy with Talos
    import talos
    import matplotlib.pyplot as plt

    r = talos.Reporting('LSTM_hidden_layer_size.csv')  # open and read csv file

    r.plot_corr()  # plot correlation matrix

    plt.show()  # show all plots


def plot_num_epochs():
    """Plot validation accuracy as function of num_epochs."""
    import csv

    import matplotlib.pyplot as plt

    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    with open('num_epochs\\LSTM_num_epochs.csv', 'r') as csvfile:  # open csv file
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
    import csv

    import matplotlib.pyplot as plt

    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    with open('batch_size\\LSTM_batch_size.csv', 'r') as csvfile:  # open csv file
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
    import csv

    import matplotlib.pyplot as plt

    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    with open('hidden_layer_size\\LSTM_hidden_layer_size.csv', 'r') as csvfile:  # open csv file
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
    import csv

    import matplotlib.pyplot as plt

    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    with open('num_cells\\LSTM_num_cells.csv', 'r') as csvfile:  # open csv file
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
    import csv

    import matplotlib.pyplot as plt

    x = []  # data that should be plotted on the x-axis (a hyperparameter)
    y = []  # data that should be plotted on the y-axis (validation accuracy)

    with open('LSTM_keep_prob.csv', 'r') as csvfile:  # open csv file
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


def get_confusion_matrix():
    """Get and plot confusion matrix (tp, fp, tn, fn) of a ML model."""
    import ml_models
    from params import hparams, config
    import os
    import main
    import logging
    import sklearn

    # get logger
    logger = logging.getLogger('RNN-SA.plotting.get_confusion_matrix')

    # create model
    model = ml_models._build_LSTM_model(hparams, config)

    # load weights
    model.load_weights(os.path.join(config['checkpoint_dir'], "weights.best.hdf5"), by_name=True)

    # compile model
    model.compile(optimizer=hparams['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

    # load data
    data = main.load_data(os.getcwd(), "panda_v3.db")

    # get loss and accuracy
    loss, accuracy = model.evaluate(data['test_X'], data['test_y'],
                                    batch_size=hparams['batch_size'],
                                    verbose=config['verbose_eval'])
    logger.info("Loss = %f, Accuracy = %f", loss, accuracy)

    # get confusion matrix
    y_pred = model.predict(data['test_X'])
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=data['test_y'], y_pred=y_pred).ravel()
    logger.info("tp = %d \nfp = %d \ntn = %d \nfn = %d", tp, fp, tn, fn)


if __name__ == "__main__":
    get_confusion_matrix()
