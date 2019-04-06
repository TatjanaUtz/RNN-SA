"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import csv
import logging
import time  # for measuring time
from random import shuffle  # for shuffle of the task-sets

import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import talos

import database_interface
import logging_config
import ml_models
import params

# default indices of all task attributes (column indices of 'Task')
DEFAULT_FEATURES = ['Task_ID', 'Priority', 'Deadline', 'Quota', 'CAPS', 'PKG', 'Arg', 'CORES',
                    'COREOFFSET',
                    'CRITICALTIME', 'Period', 'Number_of_Jobs', 'OFFSET']

# task attributes that should be used for classifying task-sets
USE_FEATURES = ['Priority', 'PKG', 'Arg', 'CRITICALTIME', 'Period', 'Number_of_Jobs']

# one hot encoding of task attribute PKG
PKG_ENCODING = {
    'cond_mod': [1, 0, 0, 0],
    'hey': [0, 1, 0, 0],
    'pi': [0, 0, 1, 0],
    'tumatmul': [0, 0, 0, 1]
}


def main():
    """Main function of project 'RNN-SA'."""
    # determine database directory and name
    db_dir, db_name = "..\\Datenbanken\\", "panda_v3.db"

    # create and initialize logger
    logger = logging_config.init_logging()

    # load the data
    logger.info("Loading and pre-processing data from the database...")
    start_time = time.time()
    data = load_data(db_dir, db_name)
    end_time = time.time()
    logger.info("Successfully loaded %d samples for training, %d samples for evaluation and %d "
                "samples for testing from the database!", len(data['train_y']), len(data['val_y']),
                len(data['test_y']))
    logger.info("Time elapsed: %f s \n", end_time - start_time)

    # hyperparameter exploration
    logger.info("Doing hyperparameter exploration...")
    start_time = time.time()
    #h = hyperparameter_exploration(data=data, name='LSTM', num='0')
    out, model = ml_models.lstm_model(data['train_X'], data['train_y'], data['val_X'],
                                      data['val_y'], params.hparams)
    end_time = time.time()
    logger.info("Finished hyperparameter exploration!")
    logger.info("Best result: ")
    logger.info("Time elapsed: %f s \n", end_time - start_time)


def hyperparameter_exploration(data, name, num):
    """Hyperparameter exploration with TALOS.

    This function explores different hyperparameter combinations with the framework TALOS.

    Args:
        data -- a dictionary with the training, testing and validation data
        name -- name of the experiment
        num -- number of the experiment
    Return:
        h -- the scan object created by TALOS with several attributes for evaluation
    """
    h = talos.Scan(
        x=data['train_X'],  # prediction features
        y=data['train_y'],  # prediction outcome variable
        params=params.hparams,  # the parameter dictionary
        model=ml_models.lstm_model,  # the Keras model as a function
        dataset_name=name,  # used for experiment log
        experiment_no=num,  # used for experiment log
        x_val=data['val_X'],  # validation data for x
        y_val=data['val_y'],  # validation data for y
        # grid_downsample=0.1,  # a float to indicate fraction for random sampling
    )
    return h


def do_plotting():
    # use filename as input
    r = talos.Reporting('alle_Ergebnisse.csv')

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


def load_data(db_dir, db_name):
    """Load the data from the database.

    Args:
        db_dir -- directory of the database
        db_name -- name of the database
    Return:
        data -- dictionary with the train, test and validation data
    """
    logger = logging.getLogger('RNN-SA.main.load_data')

    # try to create Database-object
    try:
        my_database = database_interface.Database(db_dir=db_dir, db_name=db_name)
    except ValueError as val_err:
        logger.error('Could not create Database-object: %s', val_err)
        return None

    # read table 'CorrectTaskSet'
    rows = my_database.read_table_correcttaskset()
    shuffle(rows)  # shuffle rows

    # split task-sets into task-set IDs, the task-sets (tuples of task IDs) and labels
    _, tasksets, labels = _split_tasksets(rows)

    # read table 'Task'
    task_attributes = my_database.read_table_task(convert_to_task_dict=False)

    # preprocess task attributes: delete unuseful parameters, scale data to 0 mean, unit variance
    task_attributes = _preprocess_tasks_attributes(task_attributes)

    # replace task IDs with the corresponding task attributes
    for i, taskset in enumerate(tasksets):  # iterate over all task-sets
        taskset = list(taskset)  # convert tuple to list

        # delete all tasks with Task_ID = -1
        while taskset.count(-1) > 0:
            taskset.remove(-1)

        # replace Task_ID with the corresponding task attributes
        for j, task_id in enumerate(taskset):  # iterate over all task IDs
            taskset[j] = task_attributes[task_id]

        # replace taskset in tasksets
        tasksets[i] = taskset

    # convert lists to numpy arrays
    tasksets_np = np.asarray(tasksets)
    labels_np = np.asarray(labels, np.int32)

    # pad task-sets to uniform length
    tasksets_np = _pad_sequences(tasksets_np)

    # save data shape to configuration parameters
    params.config['time_steps'] = tasksets_np.shape[1]
    params.config['element_size'] = tasksets_np.shape[2]

    data = dict()  # create empty dictionary to keep all data tidy

    # split data into training and test/validation: 80% training data, 20% test/validation data
    data["train_X"], test_val_x, data["train_y"], test_val_y = \
        sklearn.model_selection.train_test_split(tasksets_np, labels_np, test_size=0.2)

    # split test/validation in test and validation data: 50% data each, i.e. 10% of hole dataset
    data["test_X"], data["val_X"], data["test_y"], data["val_y"] = \
        sklearn.model_selection.train_test_split(test_val_x, test_val_y, test_size=0.5)

    return data


def _split_tasksets(rows):
    """Split task-sets.

    This function splits a task-set consisting of [Set_ID, Successful, TASK1_ID, TASK2_ID,
    TASK3_ID, TASK4_ID] into a list with the task-set IDs, the labels and the task IDs.

    Args:
        rows -- an array with the rows representing each a task-set
    Return:
        taskset_ids -- list with the task-set IDs
        task_ids -- list with tuples of the task IDs
        labels -- list with the labels
    """
    taskset_ids = [x[0] for x in rows]
    task_ids = [x[2:] for x in rows]
    labels = [x[1] for x in rows]
    return taskset_ids, task_ids, labels


def _preprocess_tasks_attributes(task_attributes):
    """Preprocess the task attributes.

    This function pre-processes the task attributes for machine learning:
        - unused features are deleted
        - non-numeric values are one hot encoded
        - features are normalized.

    Args:
        task_attributes -- list with the task attributes
    Return:
        task_attributes -- list with the pre-processed task attributes
    """
    features = DEFAULT_FEATURES  # get default features

    # delete unused features
    task_attributes, features = _delete_unused_features(task_attributes, features)

    # do one hot encoding for non-numeric values
    task_attributes, features = _one_hot_encoding(task_attributes, features)

    # normalize values
    task_attributes = _standardize(task_attributes)

    return task_attributes


def _delete_unused_features(task_attributes, features):
    """Delete unused features.

    This function deletes unused features. The features available are defined by features,
    the features that should be used are defined through USE_FEATURES.

    Args:
        task_attributes -- list with the task attributes
        features -- list of the available features
    Return:
        task_attributes -- list with the processed task attributes
        features -- updated list with the available features
    """
    # iterate over all default features beginning at the end of the list
    for idx, name in reversed(list(enumerate(features))):
        if name not in USE_FEATURES:  # check if feature should be deleted
            # delete feature
            task_attributes = [x[:idx] + x[idx + 1:] for x in task_attributes]

    # update features
    features = [x for x in features if x in USE_FEATURES]

    return task_attributes, features


def _one_hot_encoding(task_attributes, features):
    """Do one hot encoding.

    This function does one hot encoding for non-numeric features. Currently only the PKG feature
    is encoded. How this features should be encoded is defined through the dictionary PKG_ENCODING.

    Args:
        task_attributes -- list with the task attributes
        features -- list with the available features
    Return:
        task_attributes -- list with the processed task attributes
        features -- list with the available features
    """
    idx = features.index('PKG')  # get index of 'PKG'

    # replace PKG with one hot encoded value
    task_attributes = [x[:idx] + tuple(PKG_ENCODING[x[idx]]) + x[idx + 1:] for x in
                       task_attributes]

    # update features
    features = features[:idx] + ['PKG_cond_mod', 'PKG_hey', 'PKG_pi',
                                 'PKG_tumatmul'] + features[idx + 1:]

    return task_attributes, features


def _standardize(task_attributes):
    """Standardize the task attributes.

    This function standardizes the task attributes,. There are the following possibilities:
        - standardization: zero-mean and unit-variance
        - min-max normalization: rescaling to the range [0, 1]

    Args:
        task_attributes -- list with the task attributes
    Return:
        task_attributes -- list with the standardized/normalized task attributes
    """
    # standardization
    # standardized = sklearn.preprocessing.StandardScaler().fit_transform(task_attributes)

    # min-max normalization
    normalized = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
        task_attributes)

    # convert numpy array back to list of tuples
    # task_attributes = [tuple(x) for x in standardized]
    task_attributes = [tuple(x) for x in normalized]

    return task_attributes


def _pad_sequences(tasksets):
    """Pad sequences.

    This function pads sequences, i.e. all task-sets have the equal length of tasks.

    Args:
        tasksets -- numpy array of the task-sets [num_tasksets X num_tasks]
    Return:
        tasksets -- numpy array with the uniform task-sets [num_tasksets X max_num_tasks]

    """
    tasksets = keras.preprocessing.sequence.pad_sequences(
        sequences=tasksets,  # list of lists, where each element is a sequence
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
    return tasksets


if __name__ == "__main__":
    main()
