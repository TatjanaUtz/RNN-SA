"""Module for machine learning models with Keras."""

import logging
import os

import keras
import tensorflow as tf


def LSTM_model(x_train, y_train, x_val, y_val, hparams):
    """Keras LSTM model.

    This method builds, compiles and trains a neural network based on LSTM cells with Keras.
    The structure of this function (arguments and return parameters) must not be changed until Talos
    is used.

    Args:
        x_train -- array with features for training
        y_train -- list with labels for training
        x_val -- array with features for validation
        y_val -- list with labels for validation
        hparam -- hyperparameter dictionary
    Return:
        out -- result of the training
        model -- the Keras model
    """
    from params import config  # import configuration parameters
    logger = logging.getLogger('RNN-SA.ml_models.LSTM_model')

    # build the Keras model
    model = _build_LSTM_model(hparams, config)

    # create model for multiple GPUs if available
    try:
        parallel_model = keras.utils.multi_gpu_model(model, gpus=3, cpu_relocation=True)
        logger.info("Training using multiple GPUs...")
    except:
        parallel_model = model
        logger.info("Training using single GPU or CPU...")

    # Configure the model for training (create optimizer and loss function)
    # for binary classification the loss function should be 'binary_crossentropy'
    parallel_model.compile(
        # String (name of optimizer) or optimizer instance
        optimizer=hparams['optimizer'],
        # String (name of objective function) or objective function (default: None)
        loss='binary_crossentropy',
        # List of metrics to be evaluated by the model during training and testing; typically
        # you will use metrics=['accuracy'] (default: None)
        metrics=['accuracy'])

    # train model
    out = parallel_model.fit(
        # Numpy array of training data
        x=x_train,
        # Numpy array of target (label) data
        y=y_train,
        # Integer or None, number of samples per gradient update (default: None = 32)
        batch_size=hparams['batch_size'],
        # Integer, number of epochs to train the model; an epoch is an iteration over the entire
        # x and y data provided (default: 1)
        epochs=hparams['num_epochs'],
        # Integer, 0, 1, or 2; verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per
        # epoch (default: 1)
        verbose=config['verbose_training'],
        # List of keras.callbacks.Callback instances; list of callbacks to apply during training
        # and validation (default: None)
        callbacks=_init_callbacks(hparams, config),
        # Data on which to evaluate the loss and any model metrics at the end of each epoch; the
        # model will not be trained on this data; validation_data will override validation_split;
        # validation_data could be: tuple (x_val, y_val) of Numpy arrays or tensors, tuple
        # (x_val, y_val, val_sample_weights) of Numpy arrays, dataset or a dataset iterator; for the
        # first two cases, batch_size must be provided; for the last case, validation_steps must be
        # provided (default: None)
        validation_data=[x_val, y_val],
        # Boolean (whether to shuffle the training data before each epoch)
        shuffle=True,
    )

    return out, parallel_model


def _build_LSTM_model(hparams, config):
    # create a Sequential model
    model = keras.models.Sequential()

    # create dropout layer: applies Dropout to the input
    # Dropout consists in randomly setting a fraction rate of input units to 0 at each update
    # during training time, which helps prevent overfitting
    if hparams['keep_prob'] < 1.0:
        dropout_layer = keras.layers.Dropout(
            # float between 0 and 1, fraction of the input units to drop
            rate=1-hparams['keep_prob'],
        )

    # only one LSTM layer: layer should specify input_shape and return only the last output
    if hparams['num_cells'] == 1:
        model.add(keras.layers.LSTM(
            # positive integer, dimensionality of the output space
            units=hparams['hidden_layer_size'],
            # activation function to use; if you pass None, no activation is applied (ie. "linear"
            # activation: a(x) = x) (default: 'tanh')
            activation=hparams['hidden_activation'],
            # Boolean, whether to return the last output in the output sequence, or the full
            # sequence
            return_sequences=False,
            # expected input shape, only the first layer in a Sequential model needs to receive
            # information about its input shape
            input_shape=(config['time_steps'], config['element_size'])
        ))

        # add dropout layer if necessary
        if hparams['keep_prob'] < 1.0: model.add(dropout_layer)

    # more than one LSTM layer
    else:
        # input LSTM layer: should specify input_shape and return a sequence of outputs
        model.add(keras.layers.LSTM(
            units=hparams['hidden_layer_size'],
            activation='tanh',
            return_sequences=True,
            input_shape=(config['time_steps'], config['element_size'])))

        # add dropout layer if necessary
        if hparams['keep_prob'] < 1.0: model.add(dropout_layer)

        # more than two LSTM layers: hidden layers should return a sequence of outputs
        if hparams['num_cells'] > 2:
            for i in range(hparams['num_cells'] - 2):
                model.add(keras.layers.LSTM(
                    units=hparams['hidden_layer_size'],
                    activation='tanh',
                    return_sequences=True))

                # add dropout layer if necessary
                if hparams['keep_prob'] < 1.0: model.add(dropout_layer)

        # output LSTM layer: should return only the last output
        model.add(keras.layers.LSTM(
            units=hparams['hidden_layer_size'],
            activation='tanh',
            return_sequences=False))

        # add dropout layer if necessary
        if hparams['keep_prob'] < 1.0: model.add(dropout_layer)

    # create and add a regular densely-connected NN layer as output layer
    # for binary classification units should be 1 or 2 (number of classes, 2 if one-hot
    # encoding) and the activation should be 'sigmoid'
    model.add(keras.layers.Dense(
        # positive integer, dimensionality of the output space
        units=config['num_classes'],
        # activation function to use; if you don't specify anything, no activation is applied
        # (ie. "linear" activation: a(x) = x) (default: None)
        activation='sigmoid'))

    return model



def _init_callbacks(params, config):
    """Initialize callbacks.

    A callback is a set of functions to be applied at given stages of the training procedure.
    Callbacks can be used to get a view on internal states and statistics of the model during
    training. A list of callbacks can be passed to the fit() method of the Sequential or Model
    classes. The relevant methods of the callbacks will then be called at each stage of the
    training.
    """
    callbacks = []

    if config['use_checkpoint']:
        # create dir for checkpoints
        _create_dirs([config['checkpoint_dir']])

        # create ModelCheckpoint: save the model after every epoch
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                # path to save the model file, can contain named formatting options which will be
                # filled with the values of epoch and keys in logs (e.g. val_loss)
                filepath=os.path.join(config['checkpoint_dir'], 'weights.best.hdf5'),
                # quantity to monitor (default: 'val_loss')
                monitor='val_acc',
                verbose=config['checkpoint_verbose'],
                # verbosity mode, 0 or 1 (default: 0)
                # if True the latest best model according to the quantity monitored will not be
                # overwritten (default: False)
                save_best_only=True,
                # if True then only the model's weights will be saved, else the full model is saved
                # (default: False)
                save_weights_only=False,
                # one of [auto, min, max], if save_best_only=True the decision to overwrite the
                # current save file is made based on either the maximization or the minimization of
                # the monitored quantity, for 'val_acc' this should be max, for 'val_loss' this
                # should be min, in auto mode the direction is automatically inferred from the name
                # of the monitored quantity (default: 'auto')
                mode='auto',
                # interval (number of epochs) between checkpoints (default: 1)
                period=1
            )
        )

    if config['use_earlystopping']:
        # create EarlyStopping: stop training when a monitored quantity has stopped improving
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',  # quantity to be monitored
                # minimum change in the monitored quantity to qualify as an improvement, i.e. an
                # absolute change of less than min_delta, will count as no improvement (default: 0)
                min_delta=0,
                # number of epochs with no improvement after which training will be stopped
                # (default: 0)
                patience=0,
            )
        )

    if config['use_tensorboard']:
        # create dir for logs
        _create_dirs([config['tensorboard_log_dir']])

        # create TensorBoard: TensorBoard basic visualization, writes a log for TensorBoard
        callbacks.append(
            keras.callbacks.TensorBoard(
                # the path of the directory where to save the log files to be parsed by TensorBoard
                log_dir=config['tensorboard_log_dir'],
                # frequency (in epochs) at which to compute activation and weight histograms for the
                # layers of the model, if set to 0 histograms won't be computed, validation data
                # (or split) must be specified for histogram visualizations (default: 0)
                histogram_freq=1,
                # size of batch of inputs to feed to the network for histograms computation
                # (default: 32)
                batch_size=params['batch_size'],
                # whether to visualize the graph in TensorBoard, the log file can become quite large
                # when write_graph is set to True (default: True)
                write_graph=False,
                # whether to visualize gradient histograms in TensorBoard, histogram_freq must be
                # greater than 0 (default: False)
                write_grads=True,
                # whether to write model weights to visualize as image in TensorBoard
                # (default: False)
                write_images=False,
                # frequency (in epochs) at which selected embedding layers will be saved, if set to
                # 0 embeddings won't be computed, data to be visualized in TensorBoard's Embedding
                # tab must be passed as embeddings_data (default: 0)
                embeddings_freq=0,
                # a list of names of layers to keep eye on, if None or empty list all the embedding
                # layer will be watched (default: None)
                embeddings_layer_names=None,
                # a dictionary which maps layer name to a file name in which metadata for this
                # embedding layer is saved, in case if the same metadata file is used for all
                # embedding layers, string can be passed (default: None)
                embeddings_metadata=None,
                # data to be embedded at layers specified in embeddings_layer_names, numpy array
                # (if the model has a single input) or list of Numpy arrays (if the model has
                # multiple inputs) (default: None)
                embeddings_data=None,
                # 'batch' or 'epoch' or integer, when using 'batch' writes the losses and metrics to
                # TensorBoard after each batch, the same applies for 'epoch', if using an integer
                # (let's say 10000) the callback will write the metrics and losses to TensorBoard
                # every 10000 samples, Note that writing too frequently to TensorBoard can slow down
                # your training (default: 'epoch')
                update_freq='epoch'
            )
        )

    if config['use_reduceLR']:
        # create ReduceLROnPlateau: reduce learning rate when a metric has stopped improving
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # quantity to be monitored (default: 'val_loss')
                # factor by which the learning rate will be reduced. new_lr = lr * factor
                # (default: 0.1)
                factor=0.2,
                # number of epochs with no improvement after which learning rate will be reduced
                # (default: 10)
                patience=2,
                verbose=0,  # int, 0: quiet, 1: update messages (default: 0)
                # one of {auto, min, max}; in min mode, lr will be reduced when the quantity
                # monitored has stopped decreasing; in max mode it will be reduced when the quantity
                # monitored has stopped increasing; in auto mode, the direction is automatically
                # inferred from the name of the monitored quantity (default: 'auto')
                mode='auto',
                # threshold for measuring the new optimum, to only focus on significant changes
                # (default: 0.0001)
                min_delta=0.0001,
                # number of epochs to wait before resuming normal operation after lr has been
                # reduced (default: 0)
                cooldown=0,
                # lower bound on the learning rate (default: 0)
                min_lr=0.0001,
            )
        )

    return callbacks


def _create_dirs(dirs):
    """To create directories.

    This function creates the given directories if these directories are not found.

    Args:
        dirs -- a list of directories
    """
    # create logger
    logger = logging.getLogger('RNN-SA.util.create_dirs')

    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as exc:
        logger.error("Creating directories error: %s", exc)
