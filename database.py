"""Class and methods for database connectivity.

This module provides classes and methods for importing task-sets from the SQLite database
and formatting data into a format usable with tensorflow.
"""

import logging  # for logging
import os  # for current directory dir
import sqlite3  # for working with the database
from random import shuffle  # for shuffle of the task-sets

import numpy as np  # for arrays

TASK_ATTRIBUTES = "Task_ID, Priority, PKG, Arg, CRITICALTIME, Period, Number_of_Jobs"
TASK_PKG_DICT = {
    'cond_mod': [1, 0, 0, 0],
    'hey': [0, 1, 0, 0],
    'pi': [0, 0, 1, 0],
    'tumatmul': [0, 0, 0, 1]
}


class Database():
    """Class representing a database.

    The database is defined by the following attributes:
        db_dir -- path to the database file (*.db)
        db_name -- name of the database file (incl. .db)
    """

    def __init__(self):
        """Constructor of class Database."""
        # path to the database = current directory
        self.db_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_name = 'panda_v2.db'  # name of the database with .db extension

        self.db_connection = None  # connection to the database
        self.db_cursor = None  # cursor to work with the database

    def open_db(self):
        """Open a database.

        This method opens the database by creating a connection and a cursor.
        If no database defined by db_name can be found, an error message is logged.
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database.open_db')

        db_path = self.db_dir + "\\" + self.db_name  # full path to the database: directory + name

        if os.path.exists(db_path):  # check if database exists
            self.db_connection = sqlite3.connect(db_path)  # create connection to database
            self.db_cursor = self.db_connection.cursor()  # create cursor for database
        else:  # database does not exist
            logger.error("Database %s not found!", self.db_name)

    def close_db(self):
        """Closes a database.

        This method closes the database if a connection or cursor is saved. Prior to that the
        changes are saved.
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database.close_db')

        if self.db_connection is not None:  # check if connection exists = database open
            self.db_connection.commit()  # commit changes
            self.db_connection.close()  # close connection
            self.db_connection = None  # delete connection
            self.db_cursor = None  # delete cursor
        else:  # database already closed
            logger.debug("No open database!")

    def read_task_attributes(self):
        """Read the attributes of all tasks from the database.

        The attributes are saved in the table 'Task' in the database.
        Currently the following attributes are considered (defined in TASK_ATTRIBUTES):
            Task_ID -- id of the task
            Priority -- priority of the task
            PKG -- PKG of the task -> one-hot-encoding
            Arg -- argument of the task
            CRITICALTIME -- deadline of the task
            Period -- period of the task
            Number_of_Jobs -- number of jobs of the task
        Instead of PKG and Arg the execution time is saved (depends on PKG and Arg).

        Return:
            task_dict -- dictionary with all tasks and their attributes
                         (key = Task_ID, value = list of attributes)
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_task_attributes")

        self.open_db()  # open database

        # read all tasks
        self.db_cursor.execute("SELECT {} FROM Task".format(TASK_ATTRIBUTES))
        rows = self.db_cursor.fetchall()

        self.close_db()  # close database

        if not rows:  # no task read
            logger.debug("No task read!")
            return None

        task_dict = dict()  # create empty dictionary

        # TODO: delete task with id = -1
        task_dict[-1] = [0] * 9

        for row in rows:  # iterate over all tasks
            row = list(row)  # convert tuple to list

            # replace PKG with one-hot encoding vector
            pkg = TASK_PKG_DICT[row[2]]
            row = row[:2] + pkg + row[3:]

            task_dict[row[0]] = row[1:]  # add task to dictionary

        return task_dict

    def read_all_tasksets(self):
        """Read all task-sets from the database.

        Return:
            tasksets -- list with task-sets
            labels -- list with the labels of the task-sets
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_all_tasksets")

        self.open_db()  # open database

        # read all task-sets
        self.db_cursor.execute("SELECT * FROM TaskSet")
        rows = self.db_cursor.fetchall()

        self.close_db()  # close database

        if not rows:  # no task-set read
            logger.debug("No task-set read!")
            return None

        # shuffle rows
        shuffle(rows)

        # limit number of rows
        #rows = rows[:1000]

        tasksets = [x[2:] for x in rows]  # list with all task-sets consisting of task-ids
        labels = [x[1] for x in rows]  # list with corresponding labels

        task_attributes_dict = self.read_task_attributes()  # get dictionary with task attributes

        # replace task-ids with task attributes
        for i, taskset in enumerate(tasksets):  # iterate over all task-sets
            taskset = list(taskset)  # convert taskset-tuple to list

            # TODO: remove all tasks with id = -1
            # while taskset.count(-1) > 0:
            #     taskset.remove(-1)

            if taskset:  # at least one task is left
                # replace Task_ID with task attributes
                for j, task_id in enumerate(taskset):
                    taskset[j] = np.asarray(task_attributes_dict[task_id])

                # replace task-set in task-set list
                tasksets[i] = np.asarray(taskset)

        tasksets_np = np.asarray(tasksets, np.float32)
        labels_np = np.asarray(labels, np.int32)

        return tasksets_np, labels_np

