"""File for different plots for visualisation of the results of hyperparameter optimization.

- Single Line Plot: plot the validation accuracy as a function of one hyperparamter
- Correlation: plot correlation matrix between validation accuracy and all hyperparamters
"""

### SINGLE LINE PLOT ###
import csv

import matplotlib.pyplot as plt

x = []  # data that should be plotted on the x-axis (a hyperparameter)
y = []  # data that should be plotted on the y-axis (validation accuracy)

with open('LSTM_num_cells_1.csv', 'r') as csvfile:  # open csv file
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


### CORRELATION ###
# plot correlation between all hyperparameters and the validation accuracy with Talos
# import talos

# r = talos.Reporting('LSTM_hidden_layer_size.csv') # open and read csv file

# r.plot_corr() # plot correlation matrix

# plt.show()    # show all plots


def plot():
    hidden_size = [[0 for i in range(5)] for i in range(6)]
    num_cells = [1, 2, 3, 4, 5]

    # with open('LSTM_hidden_layers_1.csv', 'r') as csvfile:
    #     plots = csv.reader(csvfile, delimiter=',')
    #     header = next(plots)
    #     for row in plots:
    #         if int(row[9]) == 3:
    #             hidden_size[0][int(row[8]) - 1] = float(row[2])
    #         elif int(row[9]) == 9:
    #             hidden_size[1][int(row[8]) - 1] = float(row[2])
    #         elif int(row[9]) == 27:
    #             hidden_size[2][int(row[8]) - 1] = float(row[2])
    #         elif int(row[9]) == 50:
    #             hidden_size[3][int(row[8]) - 1] = float(row[2])
    #         elif int(row[9]) == 75:
    #             hidden_size[4][int(row[8]) - 1] = float(row[2])
    #         elif int(row[9]) == 100:
    #             hidden_size[5][int(row[8]) - 1] = float(row[2])

    # plt.plot(num_cells, hidden_size[0], 'bo-')
    # plt.plot(num_cells, hidden_size[1], 'go-')
    # plt.plot(num_cells, hidden_size[2], 'ro-')
    # plt.plot(num_cells, hidden_size[3], 'yo-')
    # plt.plot(num_cells, hidden_size[4], 'ko-')
    # plt.plot(num_cells, hidden_size[5], 'co-')

    # plt.legend(('hidden_size = 3', 'hidden_size = 9', 'hidden_size = 27', 'hidden_size = 50',
    #             'hidden_size = 75', 'hidden_size = 100'))


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
    plt.xticks([32, 64, 128, 256, 512, 1024])    # ticks of x-axis
    plt.yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])  # ticks of y-axis

    plt.show()  # show all plots


if __name__ == "__main__":
    plot_batch_size()
