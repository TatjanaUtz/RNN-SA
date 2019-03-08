"""Filter for database.

Only task-sets that are correct according to a exact schedulability analysis are used for training,
evaluation and testing.
The exact schedulability analysis used is simulation with SimSo.
"""

import logging
import time
from functools import reduce

from simso.configuration import Configuration
from simso.core import Model

from database_interface import Database
from logging_config import init_logging


def filter_database(database):
    """Filters a database.

    Simulation with SimSo is used to create a new table within the database that contains only valid
    task-sets according to the exact schedulability analysis.

    Args:
        database -- the database that should be filtered
    """
    # create logger
    logger = logging.getLogger('RNN-SA.database_filter.filter_database')

    # save start time
    start_time = time.time()

    # get all task-sets from the database
    taskset_ids, tasksets, labels = database.read_table_taskset()

    # get task attributes from the database
    task_attributes = database.read_table_task()

    # get the execution times from the database
    execution_times = database.read_execution_times()

    # add execution time to every task
    task_attributes = _add_execution_time(task_attributes, execution_times)

    # iterate over all tasksets
    for i, taskset in enumerate(tasksets):

        new_taskset = []  # create empty list for task-set containing the task attributes

        # iterate over all tasks
        for task_id in taskset:
            if task_id is not -1:  # valid task ID
                task = list(task_attributes[task_id])  # get task attributes and convert to list
                new_taskset.append(task)  # append task attributes to task-set

        schedulability = simulate(new_taskset)

        # compare simulation result with real result
        if schedulability is True and labels[i] == 1:  # true positive
            # Save task-set to correct table
            database.write_correct_taskset(taskset_ids[i], taskset, labels[i])

        elif schedulability is True and labels[i] == 0:  # false positive
            # save task-set to incorrect table
            logger.debug("Taskset %d - False positive", taskset_ids[i])

        elif schedulability is False and labels[i] == 1:  # false negative
            # save task-set to incorrect table
            logger.debug("Taskset %d - False negative", taskset_ids[i])

        elif schedulability is False and labels[i] == 0:  # true negative
            # save task-set to correct table
            logger.debug("Taskset %d - True negative", taskset_ids[i])
            database.write_correct_taskset(taskset_ids[i], taskset, labels[i])

        else:  # no valid combination
            logger.debug("Taskset %d - No valid combination!", taskset_ids[i])

    # log duration
    logger.info("Time elapsed for filtering: %f", time.time() - start_time)


def _add_execution_time(task_attributes, execution_times):
    """Adds the execution time to the attributes of each task.

    Args:
        task_attributes -- dictionary with task attributes
        execution_times -- dictionary with execution times

    Return:
        task_attributes -- dictionary with task attributes including the execution time
    """
    # create logger
    logger = logging.getLogger('RNN-SA.database_filter._add_execution_time')

    # iterate over all tasks
    for task_id in task_attributes:

        # convert tuple to list
        task = list(task_attributes[task_id])

        # Define execution time depending on PKG and Arg
        pkg = task_attributes[task_id][4]
        arg = task_attributes[task_id][5]
        if (pkg, arg) in execution_times:  # combination of pkg and arg exists
            execution_time = execution_times[(pkg, arg)]
        else:  # combination of pkg and arg does not exist
            # use only pkg to determine execution time
            execution_time = execution_times[pkg]

        # Add execution time to task attributes
        task.append(execution_time)

        # convert list to tuple
        task = tuple(task)

        # update task attributes dictionary
        task_attributes.update({task_id: task})

    return task_attributes


def simulate(taskset):
    """Method for simulation with SimSo.

    This method executes the simulation of a task-set. The simulation is run over the hyperperiod,
    which is the least common mean of all task periods. The task-set is schedulable if all jobs of
    all tasks in the hyperperiod can meet their deadlines.

    Args:
        taskset -- the task-set that should be analyzed
        num_tasks -- the number of tasks in this taskset
    Return:
        True -- the task-set is schedulable
        False -- the task-set is not schedulable
    """
    # create logger
    logger = logging.getLogger('RNN-SA.database_filter.simulate')

    # manual configuration: the configuration class stores all the details about a system
    configuration = Configuration()

    # Get the periods of the tasks
    periods = []
    for i in range(len(taskset)):
        if taskset[i][9] not in periods:
            periods.append(taskset[i][9])

    # Calculate the hyperperiod of the tasks
    hyper_period = _lcm(periods)

    # Define the length of simulation (= H)
    configuration.duration = hyper_period * configuration.cycles_per_ms

    # Add a property 'priority' to the task data fields
    configuration.task_data_fields['priority'] = 'int'  # 'priority' is of type int

    # Add the tasks to the list of tasks
    for i in range(len(taskset)):
        task_name = "T" + str(i)
        activation_dates = _get_activation_dates(hyper_period, taskset[i][9], taskset[i][10])
        configuration.add_task(name=task_name, identifier=i, task_type="Sporadic",
                               period=taskset[i][9], activation_date=0, wcet=taskset[i][-1],
                               deadline=taskset[i][8], list_activation_dates=activation_dates,
                               data={'priority': taskset[i][0]})

    # Add a processor to the list of processors
    configuration.add_processor(name="CPU1", identifier=1)

    # Add a scheduler:
    configuration.scheduler_info.filename = "fp_edf_scheduler.py"  # use a custom scheduler

    # Check the correctness of the configuration (without simulating it) before trying to run it
    configuration.check_all()

    # Init a model from the configuration
    model = Model(configuration)

    # Execute the simulation
    model.run_model()

    # Schedulability analysis: check for deadline miss of each job of every task
    for task in model.results.tasks:
        for job in task.jobs:
            if job.aborted:  # deadline miss
                logger.debug("%s Deadline miss!", job.name)
                return False

    return True


def _get_activation_dates(hyper_period, task_period, number_of_jobs):
    """Determine all activation dates of a task.

    This method calculates the activation dates of a task according to the three input arguments.

    Args:
        hyper_period -- the hyperperiod
        task_period -- the period of the task
        number_of_jobs -- number of the jobs = how often should the task be activated
    Return:
        activation_dates -- list of activation dates
    """
    activation_dates = []  # create empty list
    current_activation_date = 0  # initialize current activation date
    while current_activation_date <= hyper_period and len(activation_dates) < number_of_jobs:
        if current_activation_date not in activation_dates:
            activation_dates.append(current_activation_date)
        current_activation_date += task_period

    return activation_dates


def _lcm(numbers):
    """Calculate the least common multiple.

    This function calculates the least common multiple (LCM) of a list of numbers.

    Args:
        numbers -- list with the numbers
    Return:
        the least common multiple of the numbers in the given list
    """
    return reduce(lcm, numbers, 1)


def lcm(a_value, b_value):
    """Calculate the least common multiple.

    This function calculates the least common multiple (LCM) of two numbers a, b.

    Args:
        a_value, b_value -- numbers for which the LCM should be calculated
    Return:
        the least common multiple of a and b
    """
    return (a_value * b_value) // _gcd(a_value, b_value)


def _gcd(*numbers):
    """ Calculate the greatest common divisor.

    This function calculates the greatest common divisor (GCD).

    Args:
        numbers -- list with integers, for which the GCD should be calcualated
    Return:
        the greatest common divisor
    """
    from math import gcd
    return reduce(gcd, numbers)


if __name__ == "__main__":
    init_logging()
    my_db = Database()
    filter_database(my_db)
