"""Module to filter the database."""

import logging
import time

import database_interface
import logging_config
import rta


def filter_database(database):
    """Filter the database.

    This method determines all correct task-sets through an exact schedulability analysis method.
    The method used is currently response time analysis according to Audsley, as it shows better
    results and is faster than simulation. The correct task-sets are written to the table
    'CorrectTaskSet' of the database.

    Args:
        database -- a Database-object
    """
    logger = logging.getLogger('RNN-SA.database_filter.filter_database')
    logger.info('Starting to filter task-sets...')
    start_time_filter = time.time()

    # read the data-set from the database
    logger.info("Reading task-sets from the database...")
    start_time = time.time()
    dataset = database.read_table_taskset()  # read table 'TaskSet'
    end_time = time.time()
    logger.info("Read %d task-sets from the database.", len(dataset))
    logger.info("Time elapsed: %f s \n", end_time - start_time)

    # test the data-set with the response time analysis according to Audsley
    logger.info('Filtering task-sets...')
    for taskset in dataset:  # iterate over all task-sets
        schedulability = rta.rta_buttazzo(taskset)  # check schedulability of task-set
        real_result = taskset.result  # real result of the task-set

        # compare test result with real result
        if schedulability is True and real_result == 1:  # true positive
            # write correct task-set to the database
            database.write_correct_taskset(taskset)
        elif schedulability is True and real_result == 0:  # false positive
            pass
        elif schedulability is False and real_result == 1:  # false negative
            pass
        elif schedulability is False and real_result == 0:  # true negative
            # write correct task-set to the database
            database.write_correct_taskset(taskset)

    end_time = time.time()
    logger.info("Filtering of database finished!")
    logger.info("Time elapsed: %f s", end_time - start_time_filter)


if __name__ == "__main__":
    logging_config.init_logging()
    db_dir = "C:\\Users\\Tatjana\\PycharmProjects\\Datenbanken"
    db_name = "panda_v3.db"
    # try to create Database-object: table 'CorrectTaskSet' is created automatically
    try:
        my_database = database_interface.Database(db_dir=db_dir, db_name=db_name)
    except ValueError as val_err:
        logging.error('Could not create Database-object: %s', val_err)
