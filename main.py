"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""

import logging
import random
import time

random.seed(4)  # fix random seed for reproducibility

import keras
import os
# this lines are needed for systems without the python3-tk package to avoid the following errors:
# ModuleNotFoundError: No module named '_tkinter'
# Import Error: No module named '_tkinter', please install the python3-tk package
# GUI backends on Linux: Qt4Agg, GTKAgg, WXagg, TKAgg, GTK3Agg
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import numpy as np
import sklearn
import talos

import database_interface
import logging_config
import ml_models
import params
import os

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
    db_dir, db_name = os.getcwd(), "panda_v3.db"

    # create and initialize logger
    logger = logging_config.init_logging(db_dir, db_name)

    # load the data
    data = load_data(db_dir, db_name)

    ##############################################
    ### HYPERPARAMETER OPTIMIZATION WITH TALOS ###
    ##############################################
    # hyperparameter exploration
    hyperparameter_exploration(data=data, name='test', num='1')

    ##########################
    ### SINGLE KERAS MODEL ###
    ##########################
    # train and evaluate a Keras model
    # train_and_evaluate(data)


def hyperparameter_exploration(data, name, num):
    """Hyperparameter exploration with TALOS.

    This function explores different hyperparameter combinations with the framework TALOS.

    Args:
        data -- a dictionary with the training, testing and validation data
        name -- name of the experiment
        num -- number of the experiment
    """
    logger = logging.getLogger('RNN-SA.main.hyperparameter_exploration')
    logger.info("Doing hyperparameter exploration...")
    start_time = time.time()

    talos.Scan(
        x=data['train_X'],  # prediction features
        y=data['train_y'],  # prediction outcome variable
        params=params.hparams_talos,  # the parameter dictionary
        model=ml_models.LSTM_model,  # the Keras model as a function
        dataset_name=name,  # used for experiment log
        experiment_no=num,  # used for experiment log
        x_val=data['val_X'],  # validation data for x
        y_val=data['val_y'],  # validation data for y
        # grid_downsample=0.1,  # a float to indicate fraction for random sampling
        print_params=True,  # print each permutation hyperparameters
    )

    end_time = time.time()
    logger.info("Finished hyperparameter exploration!")
    logger.info("Time elapsed: %f s \n", end_time - start_time)


def train_and_evaluate(data):
    """Build, train and evaluate a Keras model.

    This function builds, trains and evaluates a Keras model with specific hyperparameters
    defined by the params.hparams dictionary.

    Args:
        data -- a dictionary with the training, testing and validation data
    """
    logger = logging.getLogger('RNN-SA.main.train_and_evaluate')
    logger.info("Training the Keras model...")
    start_time = time.time()

    # build, compile and train the Keras LSTM model
    out, model = ml_models.LSTM_model(data['train_X'], data['train_y'], data['val_X'],
                                      data['val_y'], params.hparams)

    end_time = time.time()
    logger.info("Finished training!")
    logger.info("Time elapsed: %f s \n", end_time - start_time)

    logger.info("Evaluating performance of the Keras model...")
    start_time = time.time()

    # evaluate performance of Keras model
    loss, accuracy = model.evaluate(data['test_X'], data['test_y'], batch_size=params.hparams[
        'batch_size'], verbose=params.config['verbose_eval'])
    end_time = time.time()
    logger.info("Finished evaluation!")
    logger.info("Time elapsed: %f s", end_time - start_time)
    logger.info("Loss = %f, Accuracy = %f", loss, accuracy)


def load_data(db_dir, db_name):
    """Load the data from the database.

    Args:
        db_dir -- directory of the database
        db_name -- name of the database
    Return:
        data -- dictionary with the train, test and validation data
    """
    logger = logging.getLogger('RNN-SA.main.load_data')
    logger.info("Loading and pre-processing data from the database...")
    start_time = time.time()

    # try to create Database-object
    try:
        my_database = database_interface.Database(db_dir=db_dir, db_name=db_name)
    except ValueError as val_err:
        logger.error('Could not create Database-object: %s', val_err)
        return None

    # read table 'CorrectTaskSet'
    rows = my_database.read_table_correcttaskset()
    random.shuffle(rows)  # shuffle rows

    # split task-sets into task-set IDs, the task-sets (tuples of task IDs) and labels
    tasksets, labels = _split_tasksets(rows)

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

    # pad task-sets to uniform length
    tasksets_np = _pad_sequences(tasksets)

    # convert list of labels to numpy array
    labels_np = np.asarray(labels, np.int32)

    # save data shape to configuration parameters
    params.config['time_steps'] = tasksets_np.shape[1]
    params.config['element_size'] = tasksets_np.shape[2]

    data = dict()  # create empty dictionary to keep all data tidy

    # split data into training and test/validation: 80% training data, 20% test/validation data
    data['train_X'], test_val_x, data['train_y'], test_val_y = \
        sklearn.model_selection.train_test_split(tasksets_np, labels_np, test_size=0.2,
                                                 random_state=42)

    # split test/validation in test and validation data: 50% data each, i.e. 10% of hole dataset
    data['test_X'], data['val_X'], data['test_y'], data['val_y'] = \
        sklearn.model_selection.train_test_split(test_val_x, test_val_y, test_size=0.5,
                                                 random_state=42)

    end_time = time.time()
    logger.info("Successfully loaded %d samples for training, %d samples for evaluation and %d "
                "samples for testing from the database!", len(data['train_y']), len(data['val_y']),
                len(data['test_y']))
    logger.info("Time elapsed: %f s \n", end_time - start_time)

    return data


def _split_tasksets(rows):
    """Split task-sets.

    This function splits a task-set consisting of [Set_ID, Successful, TASK1_ID, TASK2_ID, ...] into
    two lists: a list with the tuples of task IDs and a list with the labels.

    Args:
        rows -- an array with the rows representing each a task-set
    Return:
        task_ids -- list with tuples of the task IDs
        labels -- list with the labels
    """
    task_ids = [x[2:] for x in rows]
    labels = [x[1] for x in rows]
    return task_ids, labels


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
    # min-max normalization
    normalized = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
        task_attributes)
    task_attributes = [tuple(x) for x in normalized]  # convert numpy array back to list of tuples

    # standardization
    # standardized = sklearn.preprocessing.StandardScaler().fit_transform(task_attributes)
    # task_attributes = [tuple(x) for x in standardized]  # convert numpy array back to list of tuples

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
