"""Modul for machine learning models."""
import logging  # for logging
import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, LSTM, Dropout  # for input and dense Keras layers
from keras.models import Sequential  # for sequential Keras model


class BaseModel:
    """Base model for all machine learning models.

    This class defines all necessary functions of a machine learning model build with Keras.
    Source: https://github.com/Ahmkel/Keras-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, hparams, config):
        """Constructor of class BaseModel.

        Args:
            hparams -- hyperparameter
            config -- configuration parameter
        """
        self.hparams = hparams  # save hyperparameters defined in hparams.yaml
        self.config = config  # save configuration parameters defined in config.yaml
        self.model = None  # the machine learning model

        # create empty lists for callbacks, loss, accuracy, validation loss and validation accuracy
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

        # initialize callbacks
        self.init_callbacks()

    def init_callbacks(self):
        """Initialize callbacks.

        A callback is a set of functions to be applied at given stages of the training procedure.
        Callbacks can be used to get a view on internal states and statistics of the model during
        training. A list of callbacks can be passed to the fit() method of the Sequential or Model
        classes. The relevant methods of the callbacks will then be called at each stage of the
        training.

        Add saving model checkpoints and visualization with TensorBoard.
        """
        # create ModelCheckpoint: save the model after every epoch
        self.callbacks.append(
            ModelCheckpoint(
                # path to save the model file, can contain named formatting options which will be
                # filled with the values of epoch and keys in logs (e.g. val_loss)
                filepath=os.path.join(
                    self.config.checkpoint_dir,
                    '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.name),
                monitor=self.config.checkpoint_monitor,  # quantity to monitor (default: 'val_loss')
                verbose=self.config.checkpoint_verbose,  # verbosity mode, 0 or 1 (default: 0)
                # if True the latest best model according to the quantity monitored will not be
                # overwritten (default: False)
                save_best_only=self.config.checkpoint_save_best_only,
                # if True then only the model's weights will be saved, else the full model is saved
                # (default: False)
                save_weights_only=False,
                # one of [auto, min, max], if save_best_only=True the decision to overwrite the
                # current save file is made based on either the maximization or the minimization of
                # the monitored quantity, for 'val_acc' this should be max, for 'val_loss' this
                # should be min, in auto mode the direction is automatically inferred from the name
                # of the monitored quantity (default: 'auto')
                mode=self.config.checkpoint_mode,
                # interval (number of epochs) between checkpoints (default: 1)
                period=self.config.checkpoint_period
            )
        )

        # TODO: create EarlyStopping: stop training when a monitored quantity has stopped improving

        # create TensorBoard: TensorBoard basic visualization, writes a log for TensorBoard
        self.callbacks.append(
            TensorBoard(
                # the path of the directory where to save the log files to be parsed by TensorBoard
                log_dir=self.config.tensorboard_log_dir,
                # frequency (in epochs) at which to compute activation and weight histograms for the
                # layers of the model, if set to 0 histograms won't be computed, validation data
                # (or split) must be specified for histogram visualizations (default: 0)
                histogram_freq=self.config.tensorboard_histogram_freq,
                # size of batch of inputs to feed to the network for histograms computation
                # (default: 32)
                batch_size=self.hparams.batch_size,
                # whether to visualize the graph in TensorBoard, the log file can become quite large
                # when write_graph is set to True (default: True)
                write_graph=self.config.tensorboard_write_graph,
                # whether to visualize gradient histograms in TensorBoard, histogram_freq must be
                # greater than 0 (default: False)
                write_grads=self.config.tensorboard_write_grads,
                # whether to write model weights to visualize as image in TensorBoard
                # (default: False)
                write_images=self.config.tensorboard_write_images,
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
                update_freq=self.config.tensorboard_update_freq
            )
        )

        # TODO: create ReduceLROnPlateau: reduce learning rate when a metric has stopped improving

    def save(self, checkpoint_path):
        """Save a model.

        This function saves the checkpoint in the path defined in the config file.

        Args:
            checkpoint_path -- path where the checkpoint should be saved
        """
        # create logger
        logger = logging.getLogger('RNN-SA.ML_models.BaseModel.save')

        # check if model is already created
        if self.model is None:
            logger.error("No model available: you have to built the model first!")
            raise Exception("You have to build the model first.")

        # Save the models weights
        logger.debug("Saving model...")
        self.model.save_weigths(checkpoint_path)
        logger.debug("Model saved!")

    def load(self, checkpoint_path):
        """Load a model.

        This function loads the latest checkpoint from the experiment path defined in the config
        file.

        Args:
            checkpoint_path -- path where the checkpoint should be loaded from
        """
        # create logger
        logger = logging.getLogger('RNN-SA.ML_models.BaseModel.load')

        # check if model is already created
        if self.model is None:
            logger.error("No model available: you have to built the model first!")
            raise Exception("You have to built the model first.")

        # Load the models weights
        logger.debug("Loading model checkpoint % ...", checkpoint_path)
        self.model.load_weights(checkpoint_path)
        logger.debug("Model loaded!")

    def build_model(self):
        """Build a machine learning model.

        This function must be implemented within a specific model class based on the BaseModel.
        """
        # create logger
        logger = logging.getLogger('RNN-SA.ML_models.BaseModel.build_model')

        # This function must be implemented in the specific model class
        logger.error("Not implemented error!")
        raise NotImplementedError

    def train(self, train_X, train_y):
        """Train a machine learning model.

        This function trains a machine learning model build with Keras with the given training
        dataset.

        Args:
            train_X -- a input data array
            train_y -- the labels
        """
        # train the model for a given number of epochs (iterations on a dataset)
        # returns a History object, its History.history attribute is a record of training loss
        # values and metrics values at successive epochs, as well as validation loss values and
        # validation metrics values
        self.history = self.model.fit(
            # Numpy array of training data (if the model has a single input), or list of Numpy
            # arrays (if the model has multiple inputs); if input layers in the model are named, you
            # can also pass a dictionary mapping input names to Numpy arrays; x can be None if
            # feeding from framework-native tensors (e.g. TensorFlow data tensors) (default: None)
            x=train_X,
            # Numpy array of target (label) data (if the model has a single output), or list of
            # Numpy arrays (if the model has multiple outputs); if output layers in the model are
            # named, you can also pass a dictionary mapping output names to Numpy arrays; y can be
            # None if feeding from framework-native tensors (e.g. TensorFlow data tensors)
            # (default: None)
            y=train_y,
            # Integer or None, number of samples per gradient update (default: None = 32)
            batch_size=self.hparams.batch_size,
            # Integer, number of epochs to train the model; an epoch is an iteration over the entire
            # x and y data provided; Note that in conjunction with initial_epoch, epochs is to be
            # understood as "final epoch"; the model is not trained for a number of iterations given
            # by epochs, but merely until the epoch of index epochs is reached (default: 1)
            epochs=self.hparams.num_epochs,
            # Integer, 0, 1, or 2; verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per
            # epoch (default: 1)
            verbose=self.hparams.verbose_training,
            # List of keras.callbacks.Callback instances; list of callbacks to apply during training
            # and validation (default: None)
            callbacks=self.callbacks,
            # Float between 0 and 1, fraction of the training data to be used as validation data;
            # the model will set apart this fraction of the training data, will not train on it, and
            # will evaluate the loss and any model metrics on this data at the end of each epoch;
            # the validation data is selected from the last samples in the x and y data provided,
            # before shuffling (default: 0.0)
            validation_split=self.hparams.validation_split,
            # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch');
            # 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles
            # in batch-sized chunks; has no effect when steps_per_epoch is not None (default: True)
            shuffle=True,
        )

        # create model summary
        self.model.summary()

        # add training and validation loss
        self.loss.extend(self.history.history['loss'])
        self.acc.extend(self.history.history['acc'])
        self.val_loss.extend(self.history.history['val_loss'])
        self.val_acc.extend(self.history.history['val_acc'])

    def evaluate(self, test_X, test_y):
        """Evaluate a machine learning model.

        This function evaluates a machine learning model build with Keras with the test dataset.

        Args:
            test_X -- a ipnut data array
            test_y -- the labels
        """
        # evaluate a model: returns the loss value & metrics values for the model in test mode
        self.results = self.model.evaluate(
            # Numpy array of test data (if the model has a single input), or list of Numpy arrays
            # (if the model has multiple inputs); if input layers in the model are named, you can
            # also pass a dictionary mapping input names to Numpy arrays; x can be None if feeding
            # from framework-native tensors (e.g. TensorFlow data tensors) (default: None)
            x=test_X,
            # Numpy array of target (label) data (if the model has a single output), or list of
            # Numpy arrays (if the model has multiple outputs); if output layers in the model are
            # named, you can also pass a dictionary mapping output names to Numpy arrays; y can be
            # None if feeding from framework-native tensors (e.g. TensorFlow data tensors)
            # (default: None)
            y=test_y,
            # Integer or None; number of samples per evaluation step (default: None = 32)
            batch_size=self.hparams.batch_size,
            # 0 or 1, verbosity mode, 0 = silent, 1 = progress bar
            verbose=self.hparams.verbose_eval,
            # List of keras.callbacks.Callback instances; list of callbacks to apply during
            # evaluation (default: None)
            callbacks=self.callbacks
        )


class LSTMModel(BaseModel):
    """LSTM Model.

    Recurrent neural network based on LSTM cells.
    """

    def __init__(self, hparams, config):
        """Constructor of class LSTMModel.

        Args:
            hparams -- hyperparameter
            config -- configuration parameter
        """
        # Initialize BaseModel
        super(LSTMModel, self).__init__(hparams, config)

        # Build the model
        self.build_model()

    def build_model(self):
        """Build the LSTM model.

        This function creates a LSTM model.
        """
        # Create the LSTM model
        self.model = Sequential()  # create sequential model

        # create Long Short-Term Memory (LSTM) layer - Hochreiter 1997
        lstm_layer = LSTM(
            # positive integer, dimensionality of the output space
            units=self.hparams.hidden_layer_size,
            # activation function to use; if you pass None, no activation is applied (ie. "linear"
            # activation: a(x) = x) (default: 'tanh')
            activation=self.hparams.hidden_activation_function,
        )

        # create dropout layer if necessary: applies Dropout to the input
        # Dropout consists in randomly setting a fraction rate of input units to 0 at each update
        # during training time, which helps prevent overfitting
        if self.hparams.keep_prob < 1.0:
            dropout_layer = Dropout(
                # float between 0 and 1, fraction of the input units to drop
                rate=self.hparams.keep_prob
            )

        # add LSTM layers to the model
        for i in range(self.hparams.num_cells):
            self.model.add(lstm_layer)

            # add dropout layer if necessary
            if self.hparams.keep_prob < 1.0:
                self.model.add(dropout_layer)

        # create and add a regular densely-connected NN layer as output layer
        output_layer = Dense(
            # positive integer, dimensionality of the output space
            units=self.hparams.num_classes,
            # activation function to use; if you don't specify anything, no activation is applied
            # (ie. "linear" activation: a(x) = x) (default: None)
            activation='sigmoid'
        )
        self.model.add(output_layer)

        # Configure the model for training (create optimizer and loss function)
        self.model.compile(
            # String (name of optimizer) or optimizer instance
            optimizer=self.hparams.optimizer,
            # String (name of objective function) or objective function (default: None)
            loss=self.hparams.loss,
            # List of metrics to be evaluated by the model during training and testing; typically
            # you will use metrics=['accuracy'] (default: None)
            metrics=['accuracy'])
