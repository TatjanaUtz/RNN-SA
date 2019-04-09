"""Configurations for logging."""

import logging
import os


def init_logging(db_dir, db_name):
    """Initializes logging.

    Configures logging. Error messages are logged to the 'error.log' file. Info messages are logged
    to the console. The results are save in a 'result_' log file.

    Args:
        db_dir -- directory of the database, used to create file for results
        db_name -- name of the database, used to create file name for results
    """
    # create logger for traditional-SA project
    logger = logging.getLogger('RNN-SA')
    logger.setLevel(logging.INFO)

    # create file handler which logs error messages
    log_file_handler = logging.FileHandler(os.path.join(db_dir, 'error.log'), mode='w+')
    log_file_handler.setLevel(logging.ERROR)

    # create console handler with a lower log level (e.g debug or info)
    log_console_handler = logging.StreamHandler()
    log_console_handler.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_handler.setFormatter(formatter)
    log_console_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)

    return logger
