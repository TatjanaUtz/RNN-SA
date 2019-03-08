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
NUM_TASK_ATTRIBUTES = 6
TASK_PKG_DICT = {
    'cond_mod': [1, 0, 0, 0],
    'hey': [0, 1, 0, 0],
    'pi': [0, 0, 1, 0],
    'tumatmul': [0, 0, 0, 1]
}
NUM_TASK_PKGS = 4

# default task execution times
EXECUTION_TIME_DICT = {
    ("hey", 0): 1045,
    ("hey", 1000): 1094,
    ("hey", 1000000): 1071,

    ("pi", 100): 1574,
    ("pi", 10000): 1693,
    ("pi", 100000): 1870,

    ("cond_42", 41): 1350,
    ("cond_42", 42): 1376,
    ("cond_42", 10041): 1413,
    ("cond_42", 10042): 1432,
    ("cond_42", 1000041): 1368,
    ("cond_42", 1000042): 1396,

    ("cond_mod", 100): 1323,
    ("cond_mod", 103): 1351,
    ("cond_mod", 10000): 1395,
    ("cond_mod", 10003): 1391,
    ("cond_mod", 1000000): 1342,
    ("cond_mod", 1000003): 1391,

    ("tumatmul", 10): 1511,
    ("tumatmul", 11): 1543,
    ("tumatmul", 10000): 1692,
    ("tumatmul", 10001): 1662,
    ("tumatmul", 1000000): 3009,
    ("tumatmul", 1000001): 3121,

    "hey": 1070,
    "pi": 1712,
    "cond_42": 1389,
    "cond_mod": 1366,
    "tumatmul": 2090
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

    def read_task_attributes_preprocessed(self):
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

        # create task with id = -1
        task_dict[-1] = [0] * (NUM_TASK_ATTRIBUTES - 1 + NUM_TASK_PKGS)

        for row in rows:  # iterate over all tasks
            row = list(row)  # convert tuple to list

            # replace PKG with one-hot encoding vector
            pkg = TASK_PKG_DICT[row[2]]
            row = row[:2] + pkg + row[3:]

            task_dict[row[0]] = row[1:]  # add task to dictionary

        return task_dict

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

        # create task with id = -1
        task_dict[-1] = np.asarray([0] * NUM_TASK_ATTRIBUTES)

        for row in rows:  # iterate over all tasks
            row = list(row)  # convert tuple to list
            task_dict[row[0]] = np.asarray(row[1:])  # add task to dictionary

        return task_dict

    def read_all_tasksets_preprocessed(self):
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
        # rows = rows[:5]

        tasksets = [x[2:] for x in rows]  # list with all task-sets consisting of task-ids
        labels = [x[1] for x in rows]  # list with corresponding labels

        task_attributes_dict = self.read_task_attributes_preprocessed()  # get dictionary with task attributes

        seqlen_list = []

        # replace task-ids with task attributes
        for i, taskset in enumerate(tasksets):  # iterate over all task-sets
            taskset = list(taskset)  # convert taskset-tuple to list

            # TODO: remove all tasks with id = -1
            # while taskset.count(-1) > 0:
            #     taskset.remove(-1)

            if taskset:  # at least one task is left
                task_counter = 0  # number of tasks in task-set

                # replace Task_ID with task attributes
                for j, task_id in enumerate(taskset):
                    taskset[j] = np.asarray(task_attributes_dict[task_id])
                    task_counter += 1

                # replace task-set in task-set list
                tasksets[i] = np.asarray(taskset)

                # add sequence lenght to list
                seqlen_list.append(task_counter)

        tasksets_np = np.asarray(tasksets, np.float32)
        labels_np = np.asarray(labels, np.int32)
        labels_reshaped = np.reshape(labels_np, (len(labels_np), 1))
        seqlen_list_np = np.asarray(seqlen_list, np.int32)

        return tasksets_np, labels_reshaped, seqlen_list_np

    def read_all_tasksets(self):
        """Read all task-sets from the database.

        Return:
            tasksets_np -- numpy array of task-sets
            labels_np -- numpy array of the labels of the task-sets
            seqlens_np -- numpy array of the sequence lenghts (= number of tasks) of the task-sets
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
        # rows = rows[:5]

        tasksets = [x[2:] for x in rows]  # list with all task-sets consisting of task-ids
        labels = [x[1] for x in rows]  # list with corresponding labels

        task_attributes_dict = self.read_task_attributes()  # get dictionary with task attributes

        seqlens = []

        # replace task-ids with task attributes
        for i, taskset in enumerate(tasksets):  # iterate over all task-sets
            taskset = list(taskset)  # convert taskset-tuple to list

            if taskset:  # at least one task is left
                task_counter = 0  # number of tasks in task-set

                # replace Task_ID with task attributes
                for j, task_id in enumerate(taskset):
                    taskset[j] = np.asarray(task_attributes_dict[task_id])
                    if task_id is not -1:
                        task_counter += 1

                # replace task-set in task-set list
                tasksets[i] = np.asarray(taskset)

                # add sequence lenght to list
                seqlens.append(task_counter)

        tasksets_np = np.asarray(tasksets)
        labels_np = np.asarray(labels, np.int32)
        labels_np = np.reshape(labels_np, (len(labels), 1))
        seqlens_np = np.asarray(seqlens, np.int32)

        return tasksets_np, labels_np, seqlens_np

    def read_execution_times(self):
        """Read table ExecutionTimes.

        Read all rows of the table ExecutionTimes and save the columns PKG(Arg) and Average_C as
        a dictionary with
            key = PKG(Arg)
            value = Average_C.

        Return:
            execution_times -- dictionary with execution times
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_execution_times")

        self.open_db()  # open database
        # read all average execution times
        self.db_cursor.execute("SELECT [PKG(Arg)], [Average_C] FROM ExecutionTimes")
        rows = self.db_cursor.fetchall()
        self.close_db()  # close database

        # check if execution times where found
        if not rows:  # now row was read
            logger.error("Table ExecutionTimes does not exist or is empty!")

        # create dictionary with default execution times
        execution_times = EXECUTION_TIME_DICT

        # update execution time dictionary
        for row in rows:  # iterate over all rows
            # get data from row
            pkg_arg = row[0]
            average_c = row[1]

            # split pkg and arg and create dictionary entry
            if '(' in pkg_arg:  # string contains pkg and arg
                pkg, arg = pkg_arg.split('(')
                arg = int(arg[:-1])  # delete last character = ')' and format to int
                dict_entry = {(pkg, arg): average_c}
            else:  # string contains only pkg, no arg
                pkg = pkg_arg
                dict_entry = {pkg: average_c}

            # update dictionary
            execution_times.update(dict_entry)

        return execution_times

    def read_table_taskset(self):
        """Read the table TaskSet.

        Read all rows of table TaskSet and split the table in labels and a task-set containing the
        task IDs.

        Return:
            tasksets -- list of task-sets containing the task IDs
            labels -- the labels, i.e. the schedulability of the task-sets
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database_interface.read_table_taskset')

        self.open_db()  # open database
        # read all task-sets
        self.db_cursor.execute("SELECT * FROM TaskSet")
        rows = self.db_cursor.fetchall()
        self.close_db()  # close database

        if not rows:  # no task-set read
            logger.debug("No task-set read!")
            return None

        # limit number of rows
        # rows = rows[:10]

        # split taskset IDs, task-sets and labels
        taskset_ids = [x[0] for x in rows]  # list with all task-set IDs
        tasksets = [x[2:] for x in rows]  # list with all task-sets containing the task IDs
        labels = [x[1] for x in rows]  # list with corresponding labels

        return taskset_ids, tasksets, labels

    def read_table_task(self):
        """Read the table Task.

        Read all rows and columns from the table Task and save the task attributes as a dictionary
        with    key = Task_ID
                value = (Priority, Deadline, Quota, CAPS, PKG, Arg, CORES, COREOFFSET, CRITICALTIME,
                         Period, Number_of_Jobs, OFFSET).

        Return:
            task_attributes -- dictionary with the task attributes
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_table_task")

        self.open_db()  # open database
        # read all tasks
        self.db_cursor.execute("SELECT * FROM Task")
        rows = self.db_cursor.fetchall()
        self.close_db()  # close database

        if not rows:  # no task read
            logger.debug("No task read!")
            return None

        task_attributes = dict()  # create empty dictionary

        for row in rows:  # iterate over all tasks
            task_attributes[row[0]] = row[1:]  # add task to dictionary

        return task_attributes

    def write_correct_taskset(self, taskset_id, taskset, label):
        """Write a task-set to the table CorrectTaskSet.

        Args:
            taskset_id -- ID of the task-set
            taskset -- list containing the task IDs
            label -- label of the task-set = schedulabiltiy of task-set
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database_interface.write_correct_taskset')

        self.open_db()  # open database

        # create table CorrectTaskSet if it does not exist

        create_table_sql = "CREATE TABLE IF NOT EXISTS CorrectTaskSet (" \
                           "Set_ID INTEGER, " \
                           "Successful INT, " \
                           "TASK1_ID INTEGER, " \
                           "TASK2_ID INTEGER, " \
                           "TASK3_ID INTEGER, " \
                           "TASK4_ID INTEGER, " \
                           "PRIMARY KEY(Set_ID)" \
                           ");"
        try:
            self.db_cursor.execute(create_table_sql)
        except sqlite3.Error as sqle:
            logger.error(sqle)

        # sql statement for inserting or replacing a row in the CorrectTaskSet table
        insert_or_replace_sql = "INSERT OR REPLACE INTO CorrectTaskSet" \
                                "(Set_ID, Successful, TASK1_ID, TASK2_ID, TASK3_ID, TASK4_ID)" \
                                " VALUES(?, ?, ?, ?, ?, ?)"

        # insert or replace task-set
        self.db_cursor.execute(insert_or_replace_sql, (taskset_id, label, taskset[0], taskset[1],
                                                       taskset[2], taskset[3]))

        # save (commit) changes
        self.db_connection.commit()

        self.close_db()  # close database











