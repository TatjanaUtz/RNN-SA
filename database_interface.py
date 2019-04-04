"""Class and methods for database connectivity."""

import logging
import operator
import os
import sqlite3

import benchmark
import database_filter


class Task:
    """Representation of a task.

    Currently only the following attributes are integrated:
        id -- id of the task, corresponds to column 'Task_ID'
        priority -- priority of task, 1 is the highest priority
        pkg -- name of the task, corresponds to column 'PKG'
        arg -- argument of task, has influence on the execution time
        deadline -- deadline of the task
        period -- period of the task
        number_of_jobs -- number of jobs, defines how often the task is executed
        execution_time -- time needed to execute the task
    """

    def __init__(self, task_id=-1, priority=-1, pkg=None, arg=None, deadline=-1, period=-1,
                 number_of_jobs=-1, execution_time=-1):
        """Constructor"""
        self.task_id = task_id
        self.priority = priority
        self.pkg = pkg
        self.arg = arg
        self.deadline = deadline
        self.period = period
        self.number_of_jobs = number_of_jobs
        self.execution_time = execution_time
        if self.deadline == -1:
            self.deadline = self.period

    def __str__(self):
        """Represent task as string."""
        repr_str = "(id=" + str(self.task_id) + " prio=" + str(self.priority) + " " + str(self.pkg) \
                   + "(" + str(self.arg) + ") D=" + str(self.deadline) + " T=" + str(self.period) \
                   + " " + str(self.number_of_jobs) + "x C=" + str(self.execution_time) + ")"
        return repr_str


class Taskset:
    """Representation of a task-set.

    Currently only the following attributes are integrated:
        taskset_id -- ID of the task-set, corresponds to column 'Set_ID'
        result -- 1 if task-set could be successfully scheduled, otherwise 0, corresponds to column
                  'Sucessful'
        tasks -- list of tasks (of type Task)
    """

    def __init__(self, taskset_id=-1, result=-1, tasks=None):
        """Constructor."""
        self.taskset_id = taskset_id
        self.result = result
        if tasks is None:
            self.tasks = []
        else:
            self.tasks = tasks

        # Sort tasks according to priorities
        self.tasks.sort(key=operator.attrgetter('priority'))

    def __str__(self):
        """Represent Taskset object as String."""
        representation_string = "id=" + str(self.taskset_id) + " result=" + str(self.result) + " " \
                                + str([str(task) for task in self.tasks])
        return representation_string

    def __len__(self):
        """Get length of task-set = number of tasks."""
        return len(self.tasks)

    def __iter__(self):
        """Iterate over the task-list."""
        return self.tasks.__iter__()

    def __getitem__(self, index):
        """Get task at index."""
        return self.tasks[index]

    def add_task(self, task):
        """Add a new task to the task-set.

        Check the input argument. If a correct input is given, add the task to the task-set and
        sort it according to priorities.

        Args:
            task -- the task that should be added, must be of type 'Task'
        """
        # check input arguments
        if not isinstance(task, Task):  # wrong input argument
            raise ValueError("task must be of type Task")

        self.tasks.append(task)  # add task to task-set
        self.tasks.sort(
            key=operator.attrgetter('priority'))  # sort tasks according to increasing priorities


class Database:
    """Class representing a database.

    The database is defined by following attributes:
        db_dir -- path to the database file (*.db)
        db_name -- name of the database file (incl. .db)
    Additional attributes of a Database object are:
        db_connection -- connection to the database
        db_cursor -- cursor for working with the database
    """

    def __init__(self, db_dir, db_name):
        """Constructor of class Database."""

        self.db_dir = db_dir  # path to the database
        self.db_name = db_name  # name of the database
        self.db_connection = None  # connection to the database
        self.db_cursor = None  # cursor for working with the database

        # check that database exists
        self._check_if_database_exists()

        # check the database: check if all necessary tables exist
        self._check_database()

    #############################
    # check database and tables #
    #############################

    def _check_if_database_exists(self):
        """Check if the database file exists.

        This method checks if the database defined by self.db_dir and self.db_name exists. If not,
        an exception is raise.
        """
        db_path = self.db_dir + "\\" + self.db_name  # create full path to database

        # check if database exists
        if not os.path.exists(db_path):  # database doesn't exists: raise exception
            raise Exception("database '%s' not found in %s" % (self.db_name, self.db_dir,))

    def _check_database(self):
        """Check the database.

        This method checks the database, i.e. if all necessary tables are present. The necessary
        tables are
            Job
            Task
            TaskSet
            CorrectTaskSet (ExecutionTime).
        If a table does not exist in the database, it is created (if possible) or an Exception is
        raised.
        """
        # check table Job
        if not self._check_if_table_exists('Job'):  # table Job does not exist
            raise Exception("no such table: %s" % ('Job',))

        # check table Task
        if not self._check_if_table_exists('Task'):  # table Task does not exist
            raise Exception("no such table: %s" % ('Task',))

        # check table TaskSet
        if not self._check_if_table_exists('TaskSet'):  # table TaskSet does not exist
            raise Exception("no such table: %s" % ('TaskSet',))

        # check table CorrectTaskSet
        if not self._check_if_table_exists('CorrectTaskSet'):  # table CorrectTaskSet does not exist
            # check table ExecutionTime
            if not self._check_if_table_exists('ExecutionTime'):
                # table ExecutionTime does not exist: create it through benchmark
                benchmark.benchmark_execution_times(self)
                # check that table was successfully created
                if not self._check_if_table_exists('ExecutionTime'):  # something went wrong
                    raise Exception("nos such table %s - creation not possible" % ('ExecutionTime'))

            # create table CorrectTaskSet through filter
            database_filter.filter_database(self)
            # check that table was successfully created
            if not self._check_if_table_exists('CorrectTaskSet'):  # something went wrong
                raise Exception("nos such table %s - creation not possible" % ('CorrectTaskSet',))

    def _check_if_table_exists(self, table_name):
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

    def _open_db(self):
        """Open the database.

        This methods opens the database defined by self.db_dir and self.db_name by creating a
        database connection and a cursor.
        """
        db_path = self.db_dir + "\\" + self.db_name  # create full path to the database

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

    #######################
    # read / write tables #
    #######################

    def read_table_job(self, set_id=None, task_id=None):
        """Read the table Job.

        This method reads the table Job of the database. If set_id or task_id is not specified, the
        hole table is read. If set_id and task_id are specified, only the jobs of the task defined
        by set_id and task_id are read.

        Args:
            set_id -- ID of the task-set of the task which jobs should be read
            task_id -- ID of the task which jobs should be read
        Return:
            rows -- list with the job attributes
        """
        self._open_db()  # open database

        if set_id is not None and task_id is not None:
            # read all jobs of set_id and task_id
            self.db_cursor.execute("SELECT * FROM Job WHERE Set_ID = ? AND Task_ID = ?",
                                   (set_id, task_id))
        else:  # read all jobs
            self.db_cursor.execute("SELECT * FROM Job")

        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        return rows

    def read_table_task(self, task_id=None, convert_to_task_dict=True):
        """Read the table Task.

        This method reads the table Task of the database. If task_id is not specified, the hole
        table is read. If task_id is specified, only the task defined by task_id is read. The
        argument dict defines, if the task attributes are converted to a dictionary with
            key = task ID
            value = Task-object.

        Args:
            task_id -- ID of the task which should be read
            convert_to_task_dict -- whether the tasks should be returned as list or dictionary
        Return:
            rows -- list with the task attributes
            task_dict -- dictionary of the task attributes (key = task ID, value = Task-object)
        """
        self._open_db()  # open database

        if task_id is not None:  # read task with ID task_id
            self.db_cursor.execute("SELECT * FROM Task WHERE Task_ID = ?", (task_id,))
        else:  # read all tasks
            self.db_cursor.execute("SELECT * FROM Task ORDER BY Task_ID ASC")

        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        if convert_to_task_dict:  # convert task attributes to dictionary
            task_dict = self._convert_to_task_dict(rows)
            return task_dict

        return rows

    def read_table_taskset(self, taskset_id=None, task_id=None, convert=True):
        """Read the table TaskSet.

        This method reads the table TaskSet of the database. If taskset_id is specified, only the
        task-set of taskset_id is read. If task_id is specified, only the task-sets where the task
        task_id is the only task are read. If neither taskset_id nor task_id is specified, the hole
        table is read.

        Args:
            taskset_id -- ID of the task-set which should be read
            task_id -- ID of the task which should be the only task in the task-set
            convert -- whether the task-sets should be converted to objects of type Taskset
        Return:
            dataset -- list with the task-sets
        """
        self._open_db()  # open database

        if taskset_id is not None:  # read task-set with taskset_id
            self.db_cursor.execute("SELECT * FROM TaskSet WHERE Set_ID = ?", (taskset_id,))
        elif task_id is not None:  # read task-set where task_id is only task
            self.db_cursor.execute("SELECT * FROM TaskSet WHERE TASK1_ID = ? AND TASK2_ID = ? AND "
                                   "TASK3_ID = ? AND TASK4_ID = ?", (task_id, -1, -1, -1))
        else:  # read all tasks-sets
            self.db_cursor.execute("SELECT * FROM TaskSet")

        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        if convert:  # convert task-sets to objects of type Taskset
            dataset = self._convert_to_taskset(rows)
            return dataset

        return rows

    def read_table_executiontime(self, convert_to_dict=True):
        """Read the table ExecutionTime.

        This method reads the table ExecutionTime. The hole table is read, i.e. all rows.

        Args:
            convert_to_dict -- whether the execution times should be returned as list or dictionary

        Return:
            execution_times -- list with the execution times
            c_dict -- dictionary of the execution times (key = TASK_ID, value = execution
                      time)
        """
        self._open_db()  # open database

        # read all execution times
        self.db_cursor.execute("SELECT * FROM ExecutionTime")
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        if convert_to_dict:  # convert execution times to dictionary
            c_dict = self._convert_to_executiontime_dict(rows)
            return c_dict

        return rows

    def read_table_correcttaskset(self):
        """Read the table CorrectTaskSet.

        This method reads the table CorrectTaskSet of the database. If taskset_id is specified, only the task-set of taskset_id
        is read. If task_id is specified, only the task-sets where the task task_id is the only task are read. If
        neither taskset_id nor task_id is specified, the hole table is read.

        Return:
            rows -- list with the task-sets
        """
        self._open_db()  # open database

        self.db_cursor.execute("SELECT * FROM CorrectTaskSet")  # read all task-sets
        rows = self.db_cursor.fetchall()
        self._close_db()  # close database

        return rows

    def write_execution_time(self, c_dict):
        """Write the execution times to the database.

        Args:
            task_dict -- dictionary with all task execution times (key = task_id, value = execution
                         time)
        """
        # create logger
        logger = logging.getLogger('traditional-SA.database._write_execution_time')

        self._open_db()  # open database

        # create table ExecutionTime if it does not exist
        create_table_sql = "CREATE TABLE IF NOT EXISTS ExecutionTime (" \
                           "TASK_ID INTEGER, " \
                           "Average_C INTEGER, " \
                           "PRIMARY KEY(TASK_ID)" \
                           ");"
        try:
            self.db_cursor.execute(create_table_sql)
        except sqlite3.Error as sqle:
            logger.error(sqle)

        # sql statement for inserting or replacing a row in the ExecutionTime table
        insert_or_replace_sql = "INSERT OR REPLACE INTO ExecutionTime" \
                                "(TASK_ID, Average_C) VALUES(?, ?)"

        # iterate over all keys
        for key in c_dict:
            # insert or replace task-set
            self.db_cursor.execute(insert_or_replace_sql, (key, c_dict[key]))

        self._close_db()  # close database

    def write_correct_taskset(self, taskset):
        """Write the correct task-set to to the database.

        This method writes a correct taskset to the table 'CorrectTaskSet' of the database.

        Args:
            taskset -- the task-set of type Taskset that should be added to the database
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

        num_tasks = len(taskset)  # get number of tasks in the task-set

        # insert or replace task-set
        if num_tasks == 1:  # only one task
            task_id = taskset[0].task_id
            self.db_cursor.execute(insert_or_replace_sql,
                                   (taskset.taskset_id, taskset.result, taskset[0].task_id, -1, -1,
                                    -1))
        elif num_tasks == 2:  # two tasks
            self.db_cursor.execute(insert_or_replace_sql,
                                   (taskset.taskset_id, taskset.result, taskset[0].task_id,
                                    taskset[1].task_id, -1, -1))
        elif num_tasks == 3:  # three tasks
            self.db_cursor.execute(insert_or_replace_sql,
                                   (taskset.taskset_id, taskset.result, taskset[0].task_id,
                                    taskset[1].task_id, taskset[2].task_id, -1))
        elif num_tasks == 4:  # four tasks
            self.db_cursor.execute(insert_or_replace_sql,
                                   (taskset.taskset_id, taskset.result, taskset[0].task_id,
                                    taskset[1].task_id, taskset[2].task_id, taskset[3].task_id))

        self._close_db()  # close database

    ##############
    # conversion #
    ##############

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

        execution_time_dict = self.read_table_executiontime()  # read table 'ExecutionTime'

        for row in task_attributes:  # iterate over all task attribute rows
            if row[0] not in execution_time_dict:  # no execution time for task found
                raise ValueError("Could not find an execution time for task %d" % (row[0],))

            execution_time = execution_time_dict[row[0]]  # get execution time of task

            # create new task
            new_task = Task(task_id=row[0], priority=row[1], pkg=row[5], arg=row[6],
                            deadline=row[9], period=row[10], number_of_jobs=row[11],
                            execution_time=execution_time)

            # add task to dictionary
            task_dict[row[0]] = new_task

        return task_dict

    def _convert_to_taskset(self, rows):
        """Convert a list of task-sets to objects of type Taskset.

        This function converts a list of task-sets from the table TaskSet to a list of Taskset
        objects.

        Args:
            rows -- the rows read from the table TaskSet
        Return:
            dataset -- list of Taskset objects
        """
        # read table 'Task': get dictionary with task attributes
        # (key = task ID, value = Task-object)
        task_attributes = self.read_table_task()

        dataset = []  # create empty list

        # iterate over all rows
        for row in rows:
            # split taskset ID, label and taskset IDs
            taskset_id = row[0]
            label = row[1]
            task_ids = row[2:]

            # create empty task-set
            new_taskset = Taskset(taskset_id=taskset_id, result=label, tasks=[])

            # iterate over all tasks and add them to the task-set
            for task_id in task_ids:
                if task_id != -1:  # valid task-id
                    new_task = task_attributes[task_id]  # get task
                    new_taskset.add_task(new_task)  # add task to task-set

            # add task-set to dataset
            dataset.append(new_taskset)

        return dataset

    def _convert_to_executiontime_dict(self, execution_times):
        """Convert a list of execution times to a dictionary.

        This function converts a list of execution times to a dictionary with
            key = task ID
            value = execution time.

        Args:
            execution_times -- list with pure execution times
        Return:
            c_dict -- dictionary with execution times
        """
        # create dictionary with default execution times
        c_dict = dict()

        for row in execution_times:  # iterate over all execution time rows
            c_dict[row[0]] = row[1]

        return c_dict
