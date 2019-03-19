"""Utility functions helpfull within project."""

import logging  # for logging
import os


def create_dirs(dirs):
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
        logger.error("Creating directories error: %", exc)
