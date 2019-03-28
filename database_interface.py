"""Class and methods for database connectivity.

This module provides classes and methods for importing task-sets from the SQLite database
and formatting data into a format usable with tensorflow.
"""

import logging
import os
import sqlite3

from benchmark import benchmark_execution_times
from database_filter import filter_database

# default task execution times
DEFAULT_EXECUTION_TIMES = {
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
    Additional attributes of a Database object are:
        db_connection -- connection to the database
        db_cursor -- cursor for working with the database
        execution_time_dict -- dictionary with the execution times of the tasks
    """

    # TODO: identisch, aber execution time nur wenn filter
    def __init__(self):
        """Constructor of class Database."""

        self.db_dir = "C:\\Users\\tatjana.utz\\PycharmProjects\\RNN-SA"  # path to the database
        self.db_name = 'panda_v2.db'  # name of the database
        self.db_connection = None  # connection to the database
        self.db_cursor = None  # cursor for working with the database

        # initalize execution times with default values
        self.execution_time_dict = DEFAULT_EXECUTION_TIMES

        # check that database exists
        self._check_if_database_exists()

        # check the database: check if all necessary tables exist
        self._check_database()

        # read table 'ExecutionTime': update execution times
        self.execution_time_dict = self.read_table_executiontime()

    #############################
    # check database and tables #
    #############################

    # TODO: identisch
    def check_if_database_exists(self):
        """Check if the database file exists.

        This method checks if the database defined by self.db_dir and self.db_name exists. If not,
        an exception is raise.
        """
        db_path = self.db_dir + "\\" + self.db_name  # create full path to database

        # check if database exists
        if not os.path.exists(db_path):  # database doesn't exists: raise exception
            raise Exception("database '%s' not found in %s", self.db_name, self.db_dir)

    # TODO: fast identisch - Tabelle CorrectTaskSet
    def check_database(self):
        """Check the database.

        This method checks the database, i.e. if all necessary tables are present. The necessary
        tables are
            Job
            Task
            TaskSet
            ExecutionTime.
        If a table does not exist in the database, it is created (if possible) or an Exception is
        raised.
        """
        # check table Job
        if not self._check_if_table_exists('Job'):  # table Job does not exist
            raise Exception("no such table: %s", 'Job')

        # check table Task
        if not self._check_if_table_exists('Task'):  # table Task does not exist
            raise Exception("no such table: %s", 'Task')

        # check table TaskSet
        if not self._check_if_table_exists('TaskSet'):  # table TaskSet does not exist
            raise Exception("no such table: %s", 'TaslSet')

        # check table ExecutionTime
        if not self._check_if_table_exists('ExecutionTime'):
            # table ExecutionTime does not exist: create it through benchmark
            benchmark_execution_times(self)

            # check that table was successfully created
            if not self._check_if_table_exists('ExecutionTime'):  # something went wrong
                raise Exception("nos such table %s - creation not possible", 'ExecutionTime')

        # check table CorrectTaskSet
        if not self.check_if_table_exists('CorrectTaskSet'):
            # table CorrectTaskSet does not exist: create it through filter
            filter_database(self)

            # check that table was successfully created
            if not self._check_if_table_exists('CorrectTaskSet'):  # something went wrong
                raise Exception("nos such table %s - creation not possible", 'CorrectTaskSet')

        # all tables exist
        return True, None

    # TODO: identisch
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

        rows = self.db_cursor.fetchall()  # fetch all rows
        self._close_db()  # close database

        if not rows:  # no row could be fetched - table doesn't exist
            return False

        # at least one row was fetched - table exists
        return True

    #########################
    # open / close database #
    #########################

    # TODO: identisch
    def _open_db(self):
        """Open the database.

        This methods opens the database defined by self.db_dir and self.db_name by creating a
        database connection and a cursor.
        """
        db_path = self.db_dir + "\\" + self.db_name  # create full path to the database

        # create database connection and a cursor
        self.db_connection = sqlite3.connect(db_path)
        self.db_cursor = self.db_connection.cursor()

    # TODO: identisch
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

    #######################
    # read / write tables #
    #######################

    # TODO: identisch
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

        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        return rows

    # TODO: fast identisch: dictionary
    def read_table_task(self, task_id=None):
        """Read the table Task.

        This method reads the table Task of the database. If task_id is not specified, the hole
        table is read. If task_id is specified, only the task defined by task_id is read.

        Args:
            task_id -- ID of the task which should be read
            dict -- whether the tasks should be returned as list or dictionary
        Return:
            rows -- list with the task attributes
            task_dict -- dictionary of the task attributes (key = task ID, value = Task-object)
        """
        # create logger
        logger = logging.getLogger("RNN-SA.database.read_table_task")

        self._open_db()  # open database

        if task_id is not None:  # read task with task_id
            self.db_cursor.execute("SELECT * FROM Task WHERE Task_ID = ?", (task_id,))
        else:  # read all tasks
            self.db_cursor.execute("SELECT * FROM Task")

        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        return rows

    # TODO: noch nicht identisch
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

    # TODO: identisch
    def read_table_executiontime(self, dict=True):
        """Read the table ExecutionTime.

        This method reads the table ExecutionTime. The hole table is read, i.e. all rows.

        Args:
            dict -- whether the execution times should be returned as list or dictionary

        Return:
            execution_times -- list with the execution times
            c_dict -- dictionary of the execution times (key = PKG or (PKG, Arg), value = execution
                      time)
        """
        # create logger
        logger = logging.getLogger('traditional-SA.database.read_table_executiontime')

        self._open_db()  # open database

        # read all execution times
        self.db_cursor.execute("SELECT * FROM ExecutionTime")
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        if dict:  # convert execution times to dictionary
            c_dict = self._convert_to_executiontime_dict(rows)
            return c_dict

        return rows

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

    # TODO: identisch
    def write_execution_time(self, c_dict):
        """Write the execution times to the database.

        Args:
            task_dict -- dictionary with all task execution times
        """
        # create logger
        logger = logging.getLogger('RNN-SA.database_interface.write_execution_time')

        self._open_db()  # open database

        # create table ExecutionTime if it does not exist
        create_table_sql = "CREATE TABLE IF NOT EXISTS ExecutionTime (" \
                           "[PKG(Arg)] TEXT, " \
                           "Average_C INTEGER, " \
                           "PRIMARY KEY([PKG(Arg)])" \
                           ");"
        try:
            self.db_cursor.execute(create_table_sql)
        except sqlite3.Error as sqle:
            logger.error(sqle)

        # sql statement for inserting or replacing a row in the ExecutionTime table
        insert_or_replace_sql = "INSERT OR REPLACE INTO ExecutionTime" \
                                "([PKG(Arg)], Average_C)" \
                                " VALUES(?, ?)"

        # iterate over all keys
        for key in c_dict:
            if isinstance(key, str):  # key = (PKG)
                # insert or replace task-set
                self.db_cursor.execute(insert_or_replace_sql, (key, c_dict[key]))
            elif len(key) == 2:  # key = PKG(Arg)
                self.db_cursor.execute(insert_or_replace_sql,
                                       (key[0] + "(" + str(key[1]) + ")", c_dict[key]))

        self._close_db()  # close database

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

    ##############
    # conversion #
    ##############

    # TODO: identisch
    def _convert_to_task_dict(self, task_attributes):
        """Convert a list of task attributes to a dictionary of Task-objects.

        This function converts a list of task attributes to a dictionary with
            key = task ID
            value = object of type Task.

        Args:
            task_attributes -- list with pure task attributes
        Return:
            task_dict -- dictionary with Task-objects
        """
        task_dict = dict()  # create empty dictionary for tasks

        for row in task_attributes:  # iterate over all task attribute rows
            # get execution time of task depending on PKG and Arg
            pkg = row[5]
            arg = row[6]
            if (pkg, arg) in self.execution_time_dict:  # (PKG, Arg)
                execution_time = self.execution_time_dict[(pkg, arg)]
            else:  # (PKG)
                execution_time = self.execution_time_dict[pkg]

            # create new task
            new_task = Task(task_id=row[0], priority=row[1], pkg=row[5], arg=row[6],
                            deadline=row[9], period=row[10], number_of_jobs=row[11],
                            execution_time=execution_time)

            # add task to dictionary
            task_dict[row[0]] = new_task

        return task_dict

    # TODO: identisch
    def _convert_to_executiontime_dict(self, execution_times):
        """Convert a list of execution times to a dictionary.

        This function converts a list of execution times to a dictionary with
            key = PKG or (PKG, Arg)
            value = execution time.

        Args:
            execution_times -- list with pure execution times
        Return:
            c_dict -- dictionary with execution times
        """
        # create dictionary with default execution times
        c_dict = DEFAULT_EXECUTION_TIMES

        for row in execution_times:  # iterate over all execution time rows
            # extract identifier and execution time
            pkg_arg = row[0]
            average_c = row[1]

            # split pkg and arg and create new dictionary entry
            if '(' in pkg_arg:  # string contains pkg and arg
                pkg, arg = pkg_arg.split('(')
                arg = int(arg[:-1])  # delete last character = ')' and format to int
                dict_entry = {(pkg, arg): average_c}
            else:  # string contains only pkg, no arg
                pkg = pkg_arg
                dict_entry = {pkg: average_c}

            # update dictionary
            c_dict.update(dict_entry)

        return c_dict


