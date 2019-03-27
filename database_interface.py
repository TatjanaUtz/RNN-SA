"""Class and methods for database connectivity.

This module provides classes and methods for importing task-sets from the SQLite database
and formatting data into a format usable with tensorflow.
"""

import logging
import os  # for current directory dir
import sqlite3  # for working with the database
from random import shuffle  # for shuffle of the task-sets

import numpy as np  # for arrays
import sklearn  # for data preprocessing (normalization)
from sklearn.model_selection import train_test_split
from tensorflow import keras

from benchmark_runtimes import benchmark_runtimes
from database_filter import filter_database
from logging_config import init_logging

# task attributes that should be used for classifying task-sets
USE_FEATURES = ['Priority', 'PKG', 'Arg', 'CRITICALTIME', 'Period', 'Number_of_Jobs']

# one hot encoding of attribute PKG of a task
TASK_PKG_DICT = {
    'cond_mod': [1, 0, 0, 0],
    'hey': [0, 1, 0, 0],
    'pi': [0, 0, 1, 0],
    'tumatmul': [0, 0, 0, 1]
}

# default indices of all task attributes (column indices of 'Task' without Task_ID)
DEFAULT_FEATURES = ['Priority', 'Deadline', 'Quota', 'CAPS', 'PKG', 'Arg', 'CORES', 'COREOFFSET',
                    'CRITICALTIME', 'Period', 'Number_of_Jobs', 'OFFSET']

TASK_ATTRIBUTES = "Task_ID, Priority, PKG, Arg, CRITICALTIME, Period, Number_of_Jobs"
NUM_TASK_ATTRIBUTES = 6
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
        # self.db_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = 'C:\\Users\\tatjana.utz\\PycharmProjects\\RNN-SA'
        self.db_name = 'panda_v2.db'  # name of the database with .db extension

        self.db_connection = None  # connection to the database
        self.db_cursor = None  # cursor to work with the database

        # check if database exists
        if not self.check_if_database_exists():
            # database does not exist at the defined path
            raise Exception("database '{}' not found in {}".format(self.db_name, self.db_dir))

        # check database: check if all necessary tables exist
        check_value, table_name = self.check_database()
        if not check_value:
            # something is not correct with the database: at least one table is missing
            raise Exception("no such table: " + table_name)

    def _open_db(self):
        """Open the database.

        This methods opens the database defined by self.db_dir and self.db_name by creating a
        database connection and a cursor.
        """
        # create full path to the database
        db_path = self.db_dir + "\\" + self.db_name

        # create database connection and a cursor
        self.db_connection = sqlite3.connect(db_path)
        self.db_cursor = self.db_connection.cursor()

    def _close_db(self):
        """Close the database.

        This method commits the changes to the database and closes it by closing and deleting the
        database connection and the cursor.
        """
        # commit changes and close connection to the database
        self.db_connection.commit()
        self.db_connection.close()

        # delete database connection and cursor
        self.db_connection = None
        self.db_cursor = None

    def check_if_database_exists(self):
        """Check if the database file exists.

        This method checks if the database defined by self.db_dir and self.db_name exists.

        Return:
            True/False -- whether the database exists
        """
        # create full path to database
        db_path = self.db_dir + "\\" + self.db_name

        # Check if database exists
        if os.path.exists(db_path):  # database exists
            return True
        return False

    def check_database(self):
        """Check the database.

        This method checks the database, i.e. if all necessary tables are present. The necessary
        tables are
            Job
            Task
            TaskSet
            ExecutionTimes
            CorrectTaskSet
        If a table does not exist in the database, it is created (if possible).

        Return:
            True/False -- whether all necessary tables exist
            the name of the table which doesn't exist in the database
        """
        # Check table Job
        if not self.check_if_table_exists('Job'):  # table Job does not exist
            return False, 'Job'

        # Check table Task
        if not self.check_if_table_exists('Task'):  # table Task does not exist
            return False, 'Task'

        # check table TaskSet
        if not self.check_if_table_exists('TaskSet'):  # table TaskSet does not exist
            return False, 'TaskSet'

        # Check table ExecutionTimes
        if not self.check_if_table_exists('ExecutionTimes'):
            # table ExecutionTimes does not exist: create it through benchmark
            benchmark_runtimes(self)

        # check table CorrectTaskSet
        if not self.check_if_table_exists('CorrectTaskSet'):  # table CorrectTaskSet does not exist
            # Create table CorrectTaskSet
            filter_database(self)

        # all tables exist
        return True, None

    def check_if_table_exists(self, table_name):
        """Check if a table exists in the database.

        This method checks if the table defined by table_name exists in the database. This is done
        by executing a SQL query and evaluate the fetched rows. If nothing could be fetched (no rows
        available), the table doesn't exist.

        Args:
            table_name -- name of the table that should be checked
        Return:
            True/False -- whether the table exists/doesn't exist in the database
        """
        self._open_db()  # open database

        # execute the following query to determine if the table exists
        sql_query = "SELECT * from sqlite_master " \
                    "WHERE type = 'table' AND name = '{}'".format(table_name)
        self.db_cursor.execute(sql_query)

        # fetch all rows
        rows = self.db_cursor.fetchall()

        if not rows:  # no row could be fetched - table doesn't exist
            self._close_db()  # close database
            return False

        # at least one row was fetched - table exists
        self._close_db()  # close database
        return True




    def read_table_correcttaskset(self):
        """Read the table CorrectTaskSet.

        Read all rows of table CorrectTaskSet and split the table in labels and a task-set containing the
        task IDs.

        Return:
            taskset_ids -- list of IDs of the task-sets
            tasksets -- list of task-sets containing the task IDs
            labels -- the labels, i.e. the schedulability of the task-sets
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database_interface.read_table_correcttaskset')

        self._open_db()  # open database
        # read all task-sets
        self.db_cursor.execute("SELECT * FROM CorrectTaskSet")
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

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

    def read_table_executiontimes(self):
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

        self._open_db()  # open database
        # read all average execution times
        self.db_cursor.execute("SELECT [PKG(Arg)], [Average_C] FROM ExecutionTimes")
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

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

        self._open_db()  # open database
        # read all task-sets
        self.db_cursor.execute("SELECT * FROM TaskSet")
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

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

    def read_table_task(self, preprocessing=False):
        """Read the table Task.

        Read all rows and columns from the table Task and save the task attributes as an array with
            row index = Task_ID
            row content = [Priority, Deadline, Quota, CAPS, PKG, Arg, CORES, COREOFFSET, CRITICALTIME,
                         Period, Number_of_Jobs, OFFSET].

        Args:
            preprocessing -- boolean, whether preprocessing should be done or not, default: False
        Return:
            task_attributes -- array with the task attributes
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_table_task")

        self._open_db()  # open database
        # read all tasks
        self.db_cursor.execute("SELECT * FROM Task ORDER BY Task_ID ASC")
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        if not rows:  # no task read
            logger.debug("No task read!")
            return None

        # do data preprocessing
        if preprocessing:
            task_attributes = self._preprocess_tasks(rows)
        else:
            task_attributes = rows

        return task_attributes

    def read_table_job(self, task_id=None, exit_value=None):
        """Read the table Job.

        This method reads the table Job of the database. If task_id is not specified, the hole table
        is read. If task_id is specified, only the jobs of the task defined by task_id are read.
        If exit_value is specified, only the jobs with this exit_value are read.

        Args:
            task_id -- ID of the task which jobs should be read
            exit_value -- exit_value of the jobs that should be read
        Return:
            rows -- list with the job attributes
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_table_job")

        self._open_db()  # open database

        if task_id is not None and exit_value is not None:
            # read all jobs of task_id with exit_value
            self.db_cursor.execute("SELECT * FROM Job WHERE Task_ID = ? AND Exit_Value = ?",
                                   (task_id, exit_value))
        else:  # read all jobs
            self.db_cursor.execute("SELECT * FROM Job")

        rows = self.db_cursor.fetchall()  # fetch all rows
        self._close_db()  # close database

        if not rows:  # no job read
            logger.debug("No job read!")
            return None

        return rows

    def write_correct_taskset(self, taskset_id, taskset, label):
        """Write a task-set to the table CorrectTaskSet.

        Args:
            taskset_id -- ID of the task-set
            taskset -- list containing the task IDs
            label -- label of the task-set = schedulabiltiy of task-set
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database_interface.write_correct_taskset')

        self._open_db()  # open database

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

        self._close_db()  # close database

    def write_execution_time(self, task_dict):
        """Write a tuple of execution times (min, max, average) to the database.

        Args:
            task_dict -- dictionary with all task execution times (= tuple of execution times
                         (min, max, average))
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database_interface.write_execution_time')

        self._open_db()  # open database

        # create table CorrectTaskSet if it does not exist
        create_table_sql = "CREATE TABLE IF NOT EXISTS ExecutionTimes (" \
                           "[PKG(Arg)] TEXT, " \
                           "[Min_C] INTEGER, " \
                           "[Max_C] INTEGER, " \
                           "Average_C INTEGER, " \
                           "PRIMARY KEY([PKG(Arg)])" \
                           ");"
        try:
            self.db_cursor.execute(create_table_sql)
        except sqlite3.Error as sqle:
            logger.error(sqle)

        # sql statement for inserting or replacing a row in the ExecutionTime table
        insert_or_replace_sql = "INSERT OR REPLACE INTO ExecutionTimes" \
                                "([PKG(Arg)], Min_C, Max_C, Average_C)" \
                                " VALUES(?, ?, ?, ?)"

        # iterate over all keys
        for key in task_dict:
            if isinstance(key, str):  # key = (PKG)
                # insert or replace task-set
                self.db_cursor.execute(insert_or_replace_sql,
                                       (key, task_dict[key][0], task_dict[key][1],
                                        task_dict[key][2]))
            elif len(key) == 2:  # key = PKG(Arg)
                self.db_cursor.execute(insert_or_replace_sql, (
                    key[0] + "(" + str(key[1]) + ")", task_dict[key][0], task_dict[key][1],
                    task_dict[key][2]))

        # save (commit) changes
        self.db_connection.commit()
        self._close_db()  # close database

    def load_data(self):
        """Load the data from the database.

        Return:
            train_X -- array with task-sets for training
            train_y -- vector with labels for training
            test_X -- array with task-sets for test
            test_y -- vector with labels for test
        """
        # read table 'CorrectTaskSet'
        _, tasksets, labels = self.read_table_correcttaskset()

        # shuffle tasksets and labels in unisono
        data = list(zip(tasksets, labels))
        shuffle(data)
        tasksets, labels = zip(*data)

        # convert tuple to list
        tasksets = list(tasksets)

        # read table 'Task'
        task_attributes = self.read_table_task(preprocessing=True)

        # replace task IDs with the corresponding task attributes
        for i, taskset in enumerate(tasksets):  # iterate over all samples
            # convert tuple to list
            taskset = list(taskset)

            # delete all tasks with Task_ID = -1
            while taskset.count(-1) > 0:
                taskset.remove(-1)

            # replace Task_ID with task attributes
            for j, task_id in enumerate(taskset):  # iterate over all task IDs
                taskset[j] = task_attributes[task_id]

            # replace taskset in tasksets
            tasksets[i] = taskset

        # convert lists to numpy arrays
        tasksets_np = np.asarray(tasksets)
        labels_np = np.asarray(labels, np.int32)

        # pad sequences to uniform length
        tasksets_np = keras.preprocessing.sequence.pad_sequences(
            sequences=tasksets_np,  # list of lists, where each element is a sequence
            maxlen=None,  # Int, maximum length of all sequences (default: None)
            # Type of the output sequences, to pad sequences with variable length string you can use
            # object (default: 'int32')
            dtype=list,
            # String, 'pre' or 'post': pad either before or after each sequence (default: 'pre')
            padding='post',
            # String, 'pre' or 'post': remove values from sequences larger than  maxlen, either at
            # the beginning or at the end of the sequences (default: 'pre')
            truncating='pre',
            value=0.0  # Float or String, padding value (default: 0.0)
        )

        # split data into training and test/val: 80% training data, 20% test/validation data
        train_X, test_val_X, train_y, test_val_y = train_test_split(tasksets_np, labels_np, test_size=0.2)

        # split test/val in test and validation data: 50% data each
        test_X, val_X, test_y, val_y = sklearn.model_selection.train_test_split(test_val_X, test_val_y, test_size=0.5)

        # return task-sets and labels
        return train_X, train_y, test_X, test_y, val_X, val_y

    def _preprocess_tasks(self, task_attributes):
        """Preprocess the tasks.

        This function preprocessed the task attributes for machine learning.
        First all unused features are deleted. Then all non-numeric values are one hot encoded.
        Finally the features are normalized.

        Args:
            task_attributes -- dictionary with all tasks and their attributes
        """
        # --- delete unused features ---
        # filter features: only use features defined by USE_FEATURES
        task_attributes = [x[1:] for x in task_attributes]  # delete task ID
        features = DEFAULT_FEATURES  # get default features

        # iterate over all default features beginning at the end
        for idx, name in reversed(list(enumerate(features))):
            if name not in USE_FEATURES:  # check if feature should be deleted
                # delete feature
                task_attributes = [x[:idx] + x[idx + 1:] for x in task_attributes]

        # update features
        features = [x for x in features if x in USE_FEATURES]

        # --- one hot encoding ---
        # do one hot encoding for PKG feature
        idx = features.index('PKG')  # get index of 'PKG'
        # replace PKG with one hot encoded
        task_attributes = [x[:idx] + tuple(TASK_PKG_DICT[x[idx]]) + x[idx + 1:] for x in
                           task_attributes]

        # update features
        features = features[:idx] + ['PKG_cond_mod', 'PKG_hey', 'PKG_pi',
                                     'PKG_tumatmul'] + features[idx + 1:]

        # --- normalization ---
        normalized = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(task_attributes)

        # --- standardization ---
        # standardized = sklearn.preprocessing.StandardScaler().fit_transform(task_attributes)

        # convert numpy array back to list of tuples
        task_attributes = [tuple(x) for x in normalized]

        # return processed task attributes
        return task_attributes



if __name__ == "__main__":
    # initialize logging
    init_logging()

    # create database
    my_db = Database()
    tasksets, labels = my_db.load_data()
    print("Dummy")
