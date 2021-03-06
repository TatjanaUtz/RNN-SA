"""Hyperparameters and configuration parameters.

    Hyperparameters hparams_talos: a regualar Python dictionary that declares the hyperparameters
    and the boundaries, parameters may be inputted in three distinct ways:
         - as a set of discreet values in a list
         - as a range of values in a tuple (min, max, steps)
         - as a single value in a list

    Hyperparameters hparams: regular Python dictionary with static hyperparameters for Keras
    model without the usage of Talos

    Configuration parameters config: a regular Python dictionary that declares other parameters
                                     necessary to configure a Keras model
"""
import numpy as np
import keras
import os

# hyperparameter for optimization with Talos
hparams_talos = {
    ### TRAINING ###
    'batch_size': [1024, 512],
                            # number of samples per gradient update
    'num_epochs': [150],  # number of epochs to train the model (an epoch is an iteration over the
    # entire data provided)

    ### MODEL ###
    'keep_prob': [0.5],  # fraction of the input units to keep (not to drop!)
    'num_cells': [3],  # number of LSTM cells
    'hidden_layer_size': [100],   # number of neurons in
    # the LSTM layers
    'hidden_activation': [keras.activations.tanh],  # activation function to use (must be an
    # instance of Keras)

    ### COMPILE ###
    'optimizer': ['adam'],  # optimizer (must be a optimizer instance of Keras)
}

# static hyperparameter for Keras model without Talos
hparams = {
    ### TRAINING ###
    'batch_size': 128,  # number of samples per gradient update
    'num_epochs': 150,  # number of epochs to train the model (an epoch is an iteration over the
    # entire data provided)

    ### MODEL ###
    'keep_prob': 0.5,  # fraction of the input units to keep (not to drop!)
    'num_cells': 3,  # number of LSTM cells
    'hidden_layer_size': 319,   # number of neurons in the LSTM layers
    'hidden_activation': 'tanh',  # activation function to use (must be an
    # instance of Keras)

    ### COMPILE ###
    'optimizer': 'adam',  # optimizer (must be a optimizer instance of Keras)
}

# general configuration parameters
config = {
    ### CALLBACKS ###
    # ModelCheckpoint: saves the model after every epoch
    'use_checkpoint': False,  # whether to use the ModelCheckpoint callback
    'checkpoint_dir': os.path.join(os.getcwd(), "experiments", "LSTM", "checkpoints"),  # path to
         # the directory where to save the model file
    'checkpoint_verbose': 1,  # verbosity mode, 0 or 1 (default: 0)

    # EarlyStopping: stop training when a monitored quantity has stopped improving
    'use_earlystopping': True,  # whether to use the EarlyStopping callback

    # TensorBoard: TensorBoard visualization
    'use_tensorboard': True,  # whether to use the TensorBoard callback
    'tensorboard_log_dir': os.path.join(os.getcwd(), "experiments", "LSTM", "logs"),  # path to
    # the directory where to save the log files to be parsed by TensorBoard

    # ReduceLROnPlateau: reduce learning rate when a metric has stopped improving
    'use_reduceLR': True,  # whether to use the ReduceLROnPlateau callback

    ### TRAINING ###
    'verbose_training': 2,  # verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch

    ### EVALUATION ###
    'verbose_eval': 0,  # 0 or 1, verbosity mode, 0 = silent, 1 = progress bar

    ### DATA SHAPE ###
    'time_steps': 4,  # number of time steps = sequence length (= maximal number of task per
    # task-set)
    'element_size': 12,  # length of each vector in the sequence (= number of attributes per task)
    'num_classes': 1,  # number of classes, binary classification, labels are in one column
}
