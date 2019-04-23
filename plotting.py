import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

### SINGLE LINE PLOT ###
x = []  # data that should be plotted on the x-axis (a hyperparameter)
y = []  # data that should be plotted on the y-axis (validation accuracy)

with open('hidden_layer_size\\LSTM_hidden_layer_size.csv', 'r') as csvfile:  # open csv file
    plots = csv.reader(csvfile, delimiter=',')
    header = next(plots)  # read header
    for row in plots:  # iterate over all rows and read data
        x.append(int(row[9]))
        y.append(float(row[2]))


def exponential_func(x, a, b, c):
    return -a * np.exp(-b * x) + c


popt, pcov = curve_fit(exponential_func, x, y, p0=(1, 1, 1))

xx = np.linspace(0, 1000, 1000)
yy = exponential_func(xx, 0.095, 0.02, 0.977)
plt.plot(xx, yy, '--')

plt.plot(x, y, 'o')  # line plot of y as a function of x
# plt.plot([200, 200], [0, 1], 'r--')  # vertical line
# plt.plot([0, 5000], [0.977, 0.977], 'r')  # horizontal line

plt.xlabel('hidden_layer_size')
plt.ylabel('val_acc')
plt.axis([0, 1000, 0.88, 0.98])
# plt.xticks([0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

plt.show()


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

    ### CORRELATION ###
    # plot correlation between all hyperparameters and the validation accuracy with Talos
    # r = talos.Reporting('LSTM_hidden_layer_size.csv')
    # r.plot_corr()
    # plt.show()
