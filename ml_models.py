"""Modul for machine learning models."""
import logging  # for logging
import os

import keras
import tensorflow as tf  # for tensorflow
from talos.model.normalizers import \
    lr_normalizer  # necessary to include different optimizers and different degrees of learning rates
import talos


# class BaseModel:
#     """Base model for all machine learning models.
#
#     This class defines all necessary functions of a machine learning model build with Keras.
#     Source: https://github.com/Ahmkel/Keras-Project-Template/blob/master/base/base_model.py
#     """
#
#     def __init__(self, hparams, config):
#         """Constructor of class BaseModel.
#
#         Args:
#             hparams -- hyperparameter defined in params.py
#             config -- configuration parameter defined in params.py
#         """
#         self.hparams = hparams  # save hyperparameter
#         self.config = config  # save configuration parameter
#         self.model = None  # the machine learning model
#
#         # create empty lists for callbacks, loss, accuracy, validation loss and validation accuracy
#         self.callbacks = []
#         self.loss = []
#         self.acc = []
#         self.val_loss = []
#         self.val_acc = []
#
#         # initialize callbacks
#         self.init_callbacks()
#
#     def init_callbacks(self):
#         """Initialize callbacks.
#
#         A callback is a set of functions to be applied at given stages of the training procedure.
#         Callbacks can be used to get a view on internal states and statistics of the model during
#         training. A list of callbacks can be passed to the fit() method of the Sequential or Model
#         classes. The relevant methods of the callbacks will then be called at each stage of the
#         training.
#
#         Add saving model checkpoints and visualization with TensorBoard.
#         """
#         if self.config['use_checkpoint']:
#             # create ModelCheckpoint: save the model after every epoch
#             self.callbacks.append(
#                 tf.keras.callbacks.ModelCheckpoint(
#                     # path to save the model file, can contain named formatting options which will be
#                     # filled with the values of epoch and keys in logs (e.g. val_loss)
#                     filepath=os.path.join(self.config['checkpoint_dir'],
#                                           '{epoch:02d}-{val_loss:.2f}.hdf5'),
#                     # quantity to monitor (default: 'val_loss')
#                     monitor='val_loss',
#                     verbose=self.config['checkpoint_verbose'],
#                     # verbosity mode, 0 or 1 (default: 0)
#                     # if True the latest best model according to the quantity monitored will not be
#                     # overwritten (default: False)
#                     save_best_only=True,
#                     # if True then only the model's weights will be saved, else the full model is saved
#                     # (default: False)
#                     save_weights_only=True,
#                     # one of [auto, min, max], if save_best_only=True the decision to overwrite the
#                     # current save file is made based on either the maximization or the minimization of
#                     # the monitored quantity, for 'val_acc' this should be max, for 'val_loss' this
#                     # should be min, in auto mode the direction is automatically inferred from the name
#                     # of the monitored quantity (default: 'auto')
#                     mode='auto',
#                     # interval (number of epochs) between checkpoints (default: 1)
#                     period=1
#                 )
#             )
#
#         # TODO: create EarlyStopping: stop training when a monitored quantity has stopped improving
#
#         if self.config['use_tensorboard']:
#             # create TensorBoard: TensorBoard basic visualization, writes a log for TensorBoard
#             self.callbacks.append(
#                 tf.keras.callbacks.TensorBoard(
#                     # the path of the directory where to save the log files to be parsed by TensorBoard
#                     log_dir=self.config['tensorboard_log_dir'],
#                     # frequency (in epochs) at which to compute activation and weight histograms for the
#                     # layers of the model, if set to 0 histograms won't be computed, validation data
#                     # (or split) must be specified for histogram visualizations (default: 0)
#                     histogram_freq=1,
#                     # size of batch of inputs to feed to the network for histograms computation
#                     # (default: 32)
#                     batch_size=self.hparams['batch_size'],
#                     # whether to visualize the graph in TensorBoard, the log file can become quite large
#                     # when write_graph is set to True (default: True)
#                     write_graph=False,
#                     # whether to visualize gradient histograms in TensorBoard, histogram_freq must be
#                     # greater than 0 (default: False)
#                     write_grads=True,
#                     # whether to write model weights to visualize as image in TensorBoard
#                     # (default: False)
#                     write_images=False,
#                     # frequency (in epochs) at which selected embedding layers will be saved, if set to
#                     # 0 embeddings won't be computed, data to be visualized in TensorBoard's Embedding
#                     # tab must be passed as embeddings_data (default: 0)
#                     embeddings_freq=0,
#                     # a list of names of layers to keep eye on, if None or empty list all the embedding
#                     # layer will be watched (default: None)
#                     embeddings_layer_names=None,
#                     # a dictionary which maps layer name to a file name in which metadata for this
#                     # embedding layer is saved, in case if the same metadata file is used for all
#                     # embedding layers, string can be passed (default: None)
#                     embeddings_metadata=None,
#                     # data to be embedded at layers specified in embeddings_layer_names, numpy array
#                     # (if the model has a single input) or list of Numpy arrays (if the model has
#                     # multiple inputs) (default: None)
#                     embeddings_data=None,
#                     # 'batch' or 'epoch' or integer, when using 'batch' writes the losses and metrics to
#                     # TensorBoard after each batch, the same applies for 'epoch', if using an integer
#                     # (let's say 10000) the callback will write the metrics and losses to TensorBoard
#                     # every 10000 samples, Note that writing too frequently to TensorBoard can slow down
#                     # your training (default: 'epoch')
#                     update_freq='epoch'
#                 )
#             )
#
#         # TODO: create ReduceLROnPlateau: reduce learning rate when a metric has stopped improving
#
#     def save(self, checkpoint_path):
#         """Save a model.
#
#         This function saves the checkpoint in the path defined in the config file.
#
#         Args:
#             checkpoint_path -- path where the checkpoint should be saved
#         """
#         # create logger
#         logger = logging.getLogger('RNN-SA.ML_models.BaseModel.save')
#
#         # check if model is already created
#         if self.model is None:
#             logger.error("No model available: you have to built the model first!")
#             raise Exception("You have to build the model first.")
#
#         # Save the models weights
#         logger.debug("Saving model...")
#         self.model.save_weigths(checkpoint_path)
#         logger.debug("Model saved!")
#
#     def load(self, checkpoint_path):
#         """Load a model.
#
#         This function loads the latest checkpoint from the experiment path defined in the config
#         file.
#
#         Args:
#             checkpoint_path -- path where the checkpoint should be loaded from
#         """
#         # create logger
#         logger = logging.getLogger('RNN-SA.ML_models.BaseModel.load')
#
#         # check if model is already created
#         if self.model is None:
#             logger.error("No model available: you have to built the model first!")
#             raise Exception("You have to built the model first.")
#
#         # Load the models weights
#         logger.debug("Loading model checkpoint % ...", checkpoint_path)
#         self.model.load_weights(checkpoint_path)
#         logger.debug("Model loaded!")
#
#     def build_model(self):
#         """Build a machine learning model.
#
#         This function must be implemented within a specific model class based on the BaseModel.
#         """
#         # create logger
#         logger = logging.getLogger('RNN-SA.ML_models.BaseModel.build_model')
#
#         # This function must be implemented in the specific model class
#         logger.error("Not implemented error!")
#         raise NotImplementedError
#
#     def train(self, train_X, train_y):
#         """Train a machine learning model.
#
#         This function trains a machine learning model build with Keras with the given training
#         dataset.
#
#         Args:
#             train_X -- a input data array
#             train_y -- the labels
#         """
#         # train the model for a given number of epochs (iterations on a dataset)
#         # returns a History object, its History.history attribute is a record of training loss
#         # values and metrics values at successive epochs, as well as validation loss values and
#         # validation metrics values
#         self.history = self.model.fit(
#             # Numpy array of training data (if the model has a single input), or list of Numpy
#             # arrays (if the model has multiple inputs); if input layers in the model are named, you
#             # can also pass a dictionary mapping input names to Numpy arrays; x can be None if
#             # feeding from framework-native tensors (e.g. TensorFlow data tensors) (default: None)
#             x=train_X,
#             # Numpy array of target (label) data (if the model has a single output), or list of
#             # Numpy arrays (if the model has multiple outputs); if output layers in the model are
#             # named, you can also pass a dictionary mapping output names to Numpy arrays; y can be
#             # None if feeding from framework-native tensors (e.g. TensorFlow data tensors)
#             # (default: None)
#             y=train_y,
#             # Integer or None, number of samples per gradient update (default: None = 32)
#             batch_size=self.hparams['batch_size'],
#             # Integer, number of epochs to train the model; an epoch is an iteration over the entire
#             # x and y data provided; Note that in conjunction with initial_epoch, epochs is to be
#             # understood as "final epoch"; the model is not trained for a number of iterations given
#             # by epochs, but merely until the epoch of index epochs is reached (default: 1)
#             epochs=self.hparams['num_epochs'],
#             # Integer, 0, 1, or 2; verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per
#             # epoch (default: 1)
#             verbose=self.config['verbose_training'],
#             # List of keras.callbacks.Callback instances; list of callbacks to apply during training
#             # and validation (default: None)
#             callbacks=self.callbacks,
#             # Float between 0 and 1, fraction of the training data to be used as validation data;
#             # the model will set apart this fraction of the training data, will not train on it, and
#             # will evaluate the loss and any model metrics on this data at the end of each epoch;
#             # the validation data is selected from the last samples in the x and y data provided,
#             # before shuffling (default: 0.0)
#             validation_split=self.config['validation_split'],
#             # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch');
#             # 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles
#             # in batch-sized chunks; has no effect when steps_per_epoch is not None (default: True)
#             shuffle=True,
#         )
#
#         # create model summary
#         # self.model.summary()
#
#         # add training and validation loss
#         self.loss.extend(self.history.history['loss'])
#         self.acc.extend(self.history.history['acc'])
#         self.val_loss.extend(self.history.history['val_loss'])
#         self.val_acc.extend(self.history.history['val_acc'])
#
#     def evaluate(self, test_X, test_y):
#         """Evaluate a machine learning model.
#
#         This function evaluates a machine learning model build with Keras with the test dataset.
#
#         Args:
#             test_X -- a ipnut data array
#             test_y -- the labels
#         Return:
#             loss -- scalar test loss
#             accuracy -- scalar test accuracy
#         """
#         # evaluate a model: returns the loss value & metrics values for the model in test mode
#         loss, accuracy = self.model.evaluate(
#             # Numpy array of test data (if the model has a single input), or list of Numpy arrays
#             # (if the model has multiple inputs); if input layers in the model are named, you can
#             # also pass a dictionary mapping input names to Numpy arrays; x can be None if feeding
#             # from framework-native tensors (e.g. TensorFlow data tensors) (default: None)
#             x=test_X,
#             # Numpy array of target (label) data (if the model has a single output), or list of
#             # Numpy arrays (if the model has multiple outputs); if output layers in the model are
#             # named, you can also pass a dictionary mapping output names to Numpy arrays; y can be
#             # None if feeding from framework-native tensors (e.g. TensorFlow data tensors)
#             # (default: None)
#             y=test_y,
#             # Integer or None; number of samples per evaluation step (default: None = 32)
#             batch_size=self.hparams['batch_size'],
#             # 0 or 1, verbosity mode, 0 = silent, 1 = progress bar
#             verbose=self.hparams['verbose_eval'],
#         )
#
#         # return loss and accuracy
#         return loss, accuracy
#
#
# class LSTM(BaseModel):
#     """LSTM Model.
#
#     Recurrent neural network based on Long Short-Term Memory (LSTM) layers - Hochreiter 1997.
#     """
#
#     def __init__(self, hparams, config):
#         """Constructor of class LSTMModel.
#
#         Args:
#             hparams -- hyperparameter from params.py
#             config -- configuration parameter from params.py
#         """
#         # Initialize BaseModel
#         super(LSTM, self).__init__(hparams, config)
#
#         # build the model
#         self.build_model()
#
#     def build_model(self):
#         """Build the LSTM model.
#
#         This function creates a LSTM model.
#         """
#         # Create a Sequential model
#         self.model = tf.keras.models.Sequential()
#
#         # create dropout layer: applies Dropout to the input
#         # Dropout consists in randomly setting a fraction rate of input units to 0 at each update
#         # during training time, which helps prevent overfitting
#         if self.hparams['keep_prob'] < 1.0:
#             dropout_layer = tf.keras.layers.Dropout(
#                 # float between 0 and 1, fraction of the input units to drop
#                 rate=self.hparams['keep_prob'])
#
#         # only one LSTM layer: layer should specify input_shape and return only the last output
#         if self.hparams['num_cells'] == 1:
#             self.model.add(tf.keras.layers.LSTM(
#                 # positive integer, dimensionality of the output space
#                 units=self.hparams['hidden_layer_size'],
#                 # activation function to use; if you pass None, no activation is applied (ie. "linear"
#                 # activation: a(x) = x) (default: 'tanh')
#                 activation=self.hparams['hidden_activation_function'],
#                 # Boolean, whether to return the last output in the output sequence, or the full
#                 # sequence
#                 return_sequences=False,
#                 # expected input shape, only the first layer in a Sequential model needs to receive
#                 # information about its input shape
#                 input_shape=(self.config['time_steps'], self.config['element_size'])
#             ))
#
#             # add dropout layer if necessary
#             if self.hparams['keep_prob'] < 1.0: self.model.add(dropout_layer)
#
#         # more than one LSTM layer
#         else:
#             # input LSTM layer: should specify input_shape and return a sequence of outputs
#             self.model.add(tf.keras.layers.LSTM(
#                 units=self.hparams['hidden_layer_size'],
#                 activation=self.hparams['hidden_activation_function'],
#                 return_sequences=True,
#                 input_shape=(self.config['time_steps'], self.config['element_size'])))
#
#             # add dropout layer if necessary
#             if self.hparams['keep_prob'] < 1.0: self.model.add(dropout_layer)
#
#             # more than two LSTM layers: hidden layers should return a sequence of outputs
#             if self.config.num_cells > 2:
#                 for i in range(self.hparams.num_cells - 2):
#                     self.model.add(tf.keras.layers.LSTM(
#                         units=self.hparams['hidden_layer_size'],
#                         activation=self.hparams['hidden_activation_function'],
#                         return_sequences=True))
#
#                     # add dropout layer if necessary
#                     if self.hparams['keep_prob'] < 1.0: self.model.add(dropout_layer)
#
#             # output LSTM layer: should return only the last output
#             self.model.add(tf.keras.layers.LSTM(
#                 units=self.hparams['hidden_layer_size'],
#                 activation=self.hparams['hidden_activation_function'],
#                 return_sequences=False))
#
#             # add dropout layer if necessary
#             if self.hparams['keep_prob'] < 1.0: self.model.add(dropout_layer)
#
#         # create and add a regular densely-connected NN layer as output layer
#         # for binary classification units should be 1 or 2 (number of classes, 2 if one-hot
#         # encoding) and the activation should be 'sigmoid'
#         self.model.add(tf.keras.layers.Dense(
#             # positive integer, dimensionality of the output space
#             units=self.config['num_classes'],
#             # activation function to use; if you don't specify anything, no activation is applied
#             # (ie. "linear" activation: a(x) = x) (default: None)
#             activation='sigmoid'))
#
#         # Configure the model for training (create optimizer and loss function)
#         # for binary classification the loss function should be 'binary_crossentropy'
#         self.model.compile(
#             # String (name of optimizer) or optimizer instance
#             optimizer=self.hparams['optimizer'](
#                 lr_normalizer(self.hparams['learning_rate'], self.hparams['optimizer'])),
#             # String (name of objective function) or objective function (default: None)
#             loss='binary_crossentropy',
#             # List of metrics to be evaluated by the model during training and testing; typically
#             # you will use metrics=['accuracy'] (default: None)
#             metrics=['accuracy'])


def lstm_model(x_train, y_train, x_val, y_val, params):
    """LSTM model with Keras."""
    # import configuration parameter
    from params import config

    # init callbacks
    callbacks = _init_callbacks(params, config)

    # build model
    model = _build_model(params, config)

    # train model
    out = model.fit(
        # Numpy array of training data
        x=x_train,
        # Numpy array of target (label) data
        y=y_train,
        # Integer or None, number of samples per gradient update (default: None = 32)
        batch_size=params['batch_size'],
        # Integer, number of epochs to train the model; an epoch is an iteration over the entire
        # x and y data provided (default: 1)
        epochs=params['num_epochs'],
        # Integer, 0, 1, or 2; verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per
        # epoch (default: 1)
        verbose=config['verbose_training'],
        # List of keras.callbacks.Callback instances; list of callbacks to apply during training
        # and validation (default: None)
        callbacks=callbacks,
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

    return out, model


def _build_model(params, config):
    # Create a Sequential model
    model = keras.models.Sequential()

    # create dropout layer: applies Dropout to the input
    # Dropout consists in randomly setting a fraction rate of input units to 0 at each update
    # during training time, which helps prevent overfitting
    # dropout_layer = keras.layers.Dropout(
    #     # float between 0 and 1, fraction of the input units to drop
    #     rate=0)

    # only one LSTM layer: layer should specify input_shape and return only the last output
    if params['num_cells'] == 1:
        model.add(keras.layers.LSTM(
            # positive integer, dimensionality of the output space
            units=params['hidden_layer_size'],
            # activation function to use; if you pass None, no activation is applied (ie. "linear"
            # activation: a(x) = x) (default: 'tanh')
            activation='tanh',
            # Boolean, whether to return the last output in the output sequence, or the full
            # sequence
            return_sequences=False,
            # expected input shape, only the first layer in a Sequential model needs to receive
            # information about its input shape
            input_shape=(config['time_steps'], config['element_size'])
        ))

        # add dropout layer if necessary
        # if params['keep_prob'] < 1.0: model.add(dropout_layer)

    # more than one LSTM layer
    else:
        # input LSTM layer: should specify input_shape and return a sequence of outputs
        model.add(keras.layers.LSTM(
            units=params['hidden_layer_size'],
            activation='tanh',
            return_sequences=True,
            input_shape=(config['time_steps'], config['element_size'])))

        # add dropout layer if necessary
        #if params['keep_prob'] < 1.0: model.add(dropout_layer)

        # more than two LSTM layers: hidden layers should return a sequence of outputs
        if params['num_cells'] > 2:
            for i in range(params['num_cells'] - 2):
                model.add(keras.layers.LSTM(
                    units=params['hidden_layer_size'],
                    activation='tanh',
                    return_sequences=True))

                # add dropout layer if necessary
                #if params['keep_prob'] < 1.0: model.add(dropout_layer)

        # output LSTM layer: should return only the last output
        model.add(keras.layers.LSTM(
            units=params['hidden_layer_size'],
            activation='tanh',
            return_sequences=False))

        # add dropout layer if necessary
        #if params['keep_prob'] < 1.0: model.add(dropout_layer)

    # create and add a regular densely-connected NN layer as output layer
    # for binary classification units should be 1 or 2 (number of classes, 2 if one-hot
    # encoding) and the activation should be 'sigmoid'
    model.add(keras.layers.Dense(
        # positive integer, dimensionality of the output space
        units=config['num_classes'],
        # activation function to use; if you don't specify anything, no activation is applied
        # (ie. "linear" activation: a(x) = x) (default: None)
        activation='sigmoid'))

    # Configure the model for training (create optimizer and loss function)
    # for binary classification the loss function should be 'binary_crossentropy'
    model.compile(
        # String (name of optimizer) or optimizer instance
        optimizer='adam',
        # String (name of objective function) or objective function (default: None)
        loss='binary_crossentropy',
        # List of metrics to be evaluated by the model during training and testing; typically
        # you will use metrics=['accuracy'] (default: None)
        metrics=['accuracy'])

    return model


def _init_callbacks(params, config):
    """Initialize callbacks.

    A callback is a set of functions to be applied at given stages of the training procedure.
    Callbacks can be used to get a view on internal states and statistics of the model during
    training. A list of callbacks can be passed to the fit() method of the Sequential or Model
    classes. The relevant methods of the callbacks will then be called at each stage of the
    training.

    Add saving model checkpoints and visualization with TensorBoard.
    """
    callbacks = []

    if config['use_checkpoint']:
        # create ModelCheckpoint: save the model after every epoch
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                # path to save the model file, can contain named formatting options which will be
                # filled with the values of epoch and keys in logs (e.g. val_loss)
                filepath=os.path.join(config['checkpoint_dir'],
                                      '{epoch:02d}-{val_loss:.2f}.hdf5'),
                # quantity to monitor (default: 'val_loss')
                monitor='val_loss',
                verbose=config['checkpoint_verbose'],
                # verbosity mode, 0 or 1 (default: 0)
                # if True the latest best model according to the quantity monitored will not be
                # overwritten (default: False)
                save_best_only=True,
                # if True then only the model's weights will be saved, else the full model is saved
                # (default: False)
                save_weights_only=True,
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

    # TODO: create EarlyStopping: stop training when a monitored quantity has stopped improving

    if config['use_tensorboard']:
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

    # TODO: create ReduceLROnPlateau: reduce learning rate when a metric has stopped improving

    return callbacks
