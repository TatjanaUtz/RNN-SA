"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import csv
import logging  # for logging
import time  # for measuring time
from random import shuffle  # for shuffle of the task-sets

import keras
import matplotlib.pyplot as plt
import numpy as np  # for arrays
import sklearn  # for data preprocessing (normalization)
import talos as ta

import logging_config
import util
from database_interface import Database
from ml_models import lstm_model
from params import hparams, config

# default indices of all task attributes (column indices of 'Task' without Task_ID)
DEFAULT_FEATURES = ['Priority', 'Deadline', 'Quota', 'CAPS', 'PKG', 'Arg', 'CORES', 'COREOFFSET',
                    'CRITICALTIME', 'Period', 'Number_of_Jobs', 'OFFSET']

# task attributes that should be used for classifying task-sets
USE_FEATURES = ['Priority', 'PKG', 'Arg', 'CRITICALTIME', 'Period', 'Number_of_Jobs']

# one hot encoding of attribute PKG of a task
TASK_PKG_DICT = {
    'cond_mod': [1, 0, 0, 0],
    'hey': [0, 1, 0, 0],
    'pi': [0, 0, 1, 0],
    'tumatmul': [0, 0, 0, 1]
}

def main():
    """Main function."""
    logging_config.init_logging()   # create and initialize logging

    # load the data
    data = load_data()

    # save data shape
    config['time_steps'] = data["train_X"].shape[1]
    config['element_size'] = data["train_X"].shape[2]

    # create dirs for checkpoints and logs
    util.create_dirs([config['checkpoint_dir'], config['tensorboard_log_dir']])

    # ----- Talos -----
    #run_talos(train_X, train_y, val_X, val_y)
    do_plotting()
    #plot()


def run_talos(train_X, train_y, val_X, val_y):
    start_t = time.time()
    h = ta.Scan(
        x=train_X,
        y=train_y,
        params=hparams,
        dataset_name='lstm',
        experiment_no='4',
        model=lstm_model,
        #grid_downsample=0.1,
        x_val=val_X,
        y_val=val_y,
    )
    print("Time elapsed: ", time.time() - start_t)

def do_plotting():

    # use filename as input
    r = ta.Reporting('alle_Ergebnisse.csv')

    # heatmap correlation
    r.plot_corr(metric='val_acc', color_grades=5)

    # a four dimensional bar grid
    r.plot_bars(x='batch_size', y='val_acc', hue='num_cells', col='hidden_layer_size')

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

def load_data():
    """Load the data from the database.

    Return:
        data -- dictionary with the train, test and val data
    """
    # create logger
    logger = logging.getLogger('RNN-SA.main.main')

    # Create a database object
    try:
        mydb = Database()
    except Exception as exc:
        logger.error('Could not create database object: {}'.format(exc))
        return

    # read table 'CorrectTaskSet'
    _, tasksets, labels = mydb.read_table_correcttaskset()

    # shuffle tasksets and labels in unisono
    data = list(zip(tasksets, labels))
    shuffle(data)
    tasksets, labels = zip(*data)

    # convert tuple to list
    tasksets = list(tasksets)

    # read table 'Task'
    task_attributes = mydb.read_table_task()

    # preprocess task attributes: delete unuseful parameters, scale data to 0 mean, unit variance
    task_attributes = _preprocess_tasks_attributes(task_attributes)

    # replace task IDs with the corresponding task attributes
    for i, taskset in enumerate(tasksets):  # iterate over all samples
        # convert tuple to list
        taskset = list(taskset)

        # delete all tasks with Task_ID = -1
        while taskset.count(-1) > 0:
            taskset.remove(-1)

        # replace Task_ID with task attributes
        for j, task_id in enumerate(taskset):  # iterate over all task IDs
            taskset[j] = task_attributes[task_id]

        # replace taskset in tasksets
        tasksets[i] = taskset

    # convert lists to numpy arrays
    tasksets_np = np.asarray(tasksets)
    labels_np = np.asarray(labels, np.int32)

    # pad sequences to uniform length
    tasksets_np = keras.preprocessing.sequence.pad_sequences(
        sequences=tasksets_np,  # list of lists, where each element is a sequence
        maxlen=None,  # Int, maximum length of all sequences (default: None)
        # Type of the output sequences, to pad sequences with variable length string you can use
        # object (default: 'int32')
        dtype=list,
        # String, 'pre' or 'post': pad either before or after each sequence (default: 'pre')
        padding='post',
        # String, 'pre' or 'post': remove values from sequences larger than  maxlen, either at
        # the beginning or at the end of the sequences (default: 'pre')
        truncating='pre',
        value=0.0  # Float or String, padding value (default: 0.0)
    )

    # create empty dictionary to keep all data tidy
    data = dict()

    # split data into training and test/val: 80% training data, 20% test/validation data
    data["train_X"], test_val_X, data["train_y"], test_val_y = sklearn.model_selection.train_test_split(tasksets_np, labels_np, test_size=0.2)

    # split test/val in test and validation data: 50% data each
    data["test_X"], data["val_X"], data["test_y"], data["val_y"] = sklearn.model_selection.train_test_split(test_val_X, test_val_y, test_size=0.5)

    # return data dictionary
    return data

def _preprocess_tasks_attributes(task_attributes):
    """Preprocess the tasks.

    This function preprocessed the task attributes for machine learning.
    First all unused features are deleted. Then all non-numeric values are one hot encoded.
    Finally the features are normalized.

    Args:
        task_attributes -- dictionary with all tasks and their attributes
    """
    # --- delete unused features ---
    # filter features: only use features defined by USE_FEATURES
    task_attributes = [x[1:] for x in task_attributes]  # delete task ID
    features = DEFAULT_FEATURES  # get default features

    # iterate over all default features beginning at the end
    for idx, name in reversed(list(enumerate(features))):
        if name not in USE_FEATURES:  # check if feature should be deleted
            # delete feature
            task_attributes = [x[:idx] + x[idx + 1:] for x in task_attributes]

    # update features
    features = [x for x in features if x in USE_FEATURES]

    # --- one hot encoding ---
    # do one hot encoding for PKG feature
    idx = features.index('PKG')  # get index of 'PKG'
    # replace PKG with one hot encoded
    task_attributes = [x[:idx] + tuple(TASK_PKG_DICT[x[idx]]) + x[idx + 1:] for x in
                       task_attributes]

    # update features
    features = features[:idx] + ['PKG_cond_mod', 'PKG_hey', 'PKG_pi',
                                 'PKG_tumatmul'] + features[idx + 1:]

    # --- normalization ---
    normalized = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(task_attributes)

    # --- standardization ---
    # standardized = sklearn.preprocessing.StandardScaler().fit_transform(task_attributes)

    # convert numpy array back to list of tuples
    task_attributes = [tuple(x) for x in normalized]

    # return processed task attributes
    return task_attributes



if __name__ == "__main__":
    main()
