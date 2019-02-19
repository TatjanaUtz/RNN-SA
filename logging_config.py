"""Configurations for logging."""
import logging


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
