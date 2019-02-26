"""Configurations for logging."""
import logging

LOG_FILE_NAME = "results.log"


def init_logging():
    """Initializes logging.

    Configures logging. Error messages are logged to the 'error.log' file. All messages are logged
    to the console.
    """
    # create logger for 'RNN-SA' project
    logger = logging.getLogger('RNN-SA')
    logger.setLevel(logging.DEBUG)

    # create file handler which logs error messages
    log_file_handler = logging.FileHandler('error.log', mode='w+')
    log_file_handler.setLevel(logging.ERROR)

    # create console handler with a lower log level (e.g. debug or info)
    log_console_handler = logging.StreamHandler()
    log_console_handler.setLevel(logging.DEBUG)

    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_handler.setFormatter(formatter)
    log_console_handler.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)

    return logger

def log_results(name):
    """Log results of a recurrent neural network.

    Overview of the results of a recurrent neural network is printed.

    Args:
        name -- name of the recurrent network
    """
    # create logger
    logger = logging.getLogger('RNN-SA.logging_config.py.log_results')

    # check input arguments
    if not isinstance(name, str):   # invalid argument for name
        logger.error("Invalid argument for 'name': is %s but must be string!", type(name))
        return

    # log results to a file
    log_file = open(LOG_FILE_NAME, 'a+')
    log_file.writable("\n")
    title_string = "---------- " + name + " ----------"
    log_file.writable(title_string + "\n")
    # TODO: log results and hyperparameter
    log_file.write("-" * len(title_string) + "\n")

    # Print results to console
    logger.info(title_string)
    # TODO: print results
    logger.info("-" * len(title_string))
