"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""
from database import Database
from logging_config import init_logging


def main():
    """Main function of project.

    Run this method to perform schedulability analysis wit recurrent neural network (RNN).
    """
    logger = init_logging()  # initialize logging

    my_database = Database()
    tasksets, labels = my_database.read_all_tasksets()
    logger.debug("Tasksets = %s", str(tasksets))
    logger.debug("Labels = %s", str(labels))


if __name__ == "__main__":
    main()
