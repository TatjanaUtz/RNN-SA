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

hparams = {
    ### TRAINING ###
    'batch_size': [100],  # size of each batch of data that is feed into the model
    'num_epochs': [150],  # number of iterations to run the dataset through the model

    ### MODEL ###
    # 'keep_prob': [1.0],  # float between 0 and 1, fraction of the input units to drop
    'num_cells': [1],  # number of LSTM cells
    'hidden_layer_size': [27],
    # size of hidden dimension, 3 times the amount of element_size
    # 'hidden_activation_function': ['tanh'],  # activation function to use; if you pass None, no
    # activation is applied (ie. "linear" activation: a(x) = x) (default: 'tanh')

    ### COMPILE ###
    # 'optimizer': [keras.optimizers.Adam],  # String (name of optimizer) or optimizer instance
    # 'learning_rate': [0.0001],  # float >= 0, learning rate
}

config = {
    ### GPU support ###
    'use_gpu': True,    # whether to use GPU(s) for training
    
    ### CALLBACKS ###
    # ModelCheckpoint: saves the model after every epoch
    'use_checkpoint': False,  # whether to use the ModelCheckpoint callback
    'checkpoint_dir': ".\\experiments\\LSTM\\checkpoints\\",  # path to save the model file
    'checkpoint_verbose': 1,  # verbosity mode, 0 or 1 (default: 0)

    # EarlyStopping: stop training when a monitored quantity has stopped improving
    'use_earlystopping': True,  # whether to use the EarlyStopping callback

    # TensorBoard: TensorBoard visualization
    'use_tensorboard': False,  # whether to use the TensorBoard callback
    'tensorboard_log_dir': '.\\experiments\\LSTM\\logs\\',  # the path of the directory where
    # to save the log files to be parsed by Tensorflow

    # ReduceLROnPlateau: reduce learning rate when a metric has stopped improving
    'use_reduceLR': True,  # whether to use the ReduceLROnPlateau callback

    ### TRAINING ###
    'verbose_training': 2,  # Integer, 0, 1, or 2; verbosity mode, 0 = silent, 1 = progress bar,
    # 2 = one line per epoch (default: 1)

    ### EVALUATION ###
    'verbose_eval': 0,  # 0 or 1, verbosity mode, 0 = silent, 1 = progress bar

    ### DATA ###
    'time_steps': 4,  # number of time steps = sequence length (= maximal number of task per
    # task-set)
    'element_size': 12,  # length of each vector in the sequence (= number of attributes per task)
    'num_classes': 1,  # number of classes, binary classification, labels are in one column
}
