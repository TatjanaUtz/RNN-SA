"""Hyperparameters and configuration parameters.

    Hyperparameters hparams: a regualar Python dictionary that declares the hyperparameters and the
                             boundaries
                             parameters may be inputted in three distinct ways:
                             - as a set of discreet values in a list
                             - as a range of values in a tuple (min, max, steps)
                             - as a single value in a list

    Configuration parameters config: a regular Python dictionary that declares other parameters
                                     necessary to configure a Keras model
"""
import keras

hparams = {
    ### TRAINING ###
    'batch_size': [10, 100],  # size of each batch of data that is feed into the model
    'num_epochs': [1, 10],  # number of iterations to run the dataset through the model

    ### MODEL ###
    'keep_prob': [1.0, 0.5],  # float between 0 and 1, fraction of the input units to drop
    'num_cells': [1, 2, 3],  # number of LSTM cells
    'hidden_layer_size': [9, 27, 100],
# size of hidden dimension, 3 times the amount of element_size
    'hidden_activation_function': ['tanh', 'relu', 'elu'],  # activation function to use; if you pass None, no
    # activation is applied (ie. "linear" activation: a(x) = x) (default: 'tanh')

    ### COMPILE ###
    'optimizer': [keras.optimizers.Adam, keras.optimizers.RMSprop, keras.optimizers.Nadam],  # String (name of optimizer) or optimizer instance
    'learning_rate': (0.0001, 0.5, 10),  # float >= 0, learning rate
}

config = {
    ### CALLBACKS ###
    # ModelCheckpoint: saves the model after every epoch
    'use_checkpoint': False,  # whether to use the ModelCheckpoint callback
    'checkpoint_dir': ".\\experiments\\LSTM\\checkpoints\\",  # path to save the model file
    'checkpoint_verbose': 1,  # verbosity mode, 0 or 1 (default: 0)

    # TensorBoard: TensorBoard visualization
    'use_tensorboard': False,  # wheter to use the TensorBoard callback
    'tensorboard_log_dir': '.\\experiments\\LSTM\\logs\\test\\',  # the path of the directory where
    # to save the log files to be parsed by Tensorflow

    ### TRAINING ###
    'verbose_training': 0,  # Integer, 0, 1, or 2; verbosity mode, 0 = silent, 1 = progress bar,
    # 2 = one line per epoch (default: 1)
    'validation_split': 0.25,  # fraction of the training data to be used as validation data

    ### EVALUATION ###
    'verbose_eval': 0,  # 0 or 1, verbosity mode, 0 = silent, 1 = progress bar

    ### DATA ###
    'time_steps': 4,  # number of time steps = sequence length (= maximal number of task per
    # task-set)
    'element_size': 12,  # length of each vector in the sequence (= number of attributes per task)
    'num_classes': 1,  # number of classes, binary classification, labels are in one column
}
