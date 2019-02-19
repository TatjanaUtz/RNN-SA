"""Class and methods for database connectivity.

This module provides classes and methods for importing task-sets from the SQLite database
and formatting data into a format usable with tensorflow.
"""

import logging  # for logging
import os  # for current directory dir
import sqlite3  # for working with the database

TASK_ATTRIBUTES = "Task_ID, Priority, PKG, Arg, CRITICALTIME, Period, Number_of_Jobs"


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

        # limit number of rows
        rows = rows[:10]

        tasksets = [x[2:] for x in rows]  # list with all task-sets consisting of task-ids
        labels = [x[1] for x in rows]  # list with corresponding labels

        task_attributes_dict = self.read_task_attributes()  # get dictionary with task attributes

        # replace task-ids with task attributes
        for taskset in tasksets:    # iterate over all task-sets
            for i in range(len(taskset)):    # iterate over all tasks
                if taskset[i] is not -1:   # valid task-id
                    taskset[i] = 99

            print(taskset)


        return tasksets, labels

    def read_task_attributes(self):
        """Read the attributes of all tasks from the database.

        The attributes are saved in the table 'Task' in the database.
        Currently the following attributes are considered (defined in TASK_ATTRIBUTES):
            Task_ID -- id of the task
            Priority -- priority of the task
            PKG -- PKG of the task
            Arg -- argument of the task
            CRITICALTIME -- deadline of the task
            Period -- period of the task
            Number_of_Jobs -- number of jobs of the task
        Instead of PKG and Arg the execution time is saved (depends on PKG and Arg).

        Return:
            task_dict -- dictionary with all tasks and their attributes
                         (key = Task_ID, value = tuple of attributes)
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

        for row in rows:  # iterate over all tasks
            task_dict[row[0]] = row[1:]  # add task to dictionary

        return task_dict
