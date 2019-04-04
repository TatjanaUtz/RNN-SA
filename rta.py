"""Response Time Analysis.

Response Time Analysis Methods:
    rta_audsley: RTA with start value according to Audsley.
    rta_buttazzo: RTA with start value according to Buttazzo.
The methods only differ in the starting value for response time calculation.
"""
import logging

import math

import database_interface


def rta_audsley(taskset):
    """Response Time Analysis according to Audsley.

    Check the schedulability of a task-set with response time analysis.
    Calculate the response times of all tasks. The task-set is schedulable if and only if for all
    tasks: R_i <= D_i

    Keyword arguments:
        taskset -- the task-set that should be tested
    Return value:
        True/False -- schedulability of task-set
    """
    # Check input argument: must be a TaskSet
    if not isinstance(taskset, database_interface.Taskset):  # Invalid input argument
        raise ValueError("taskset must be of type Taskset")

    # Check schedulability of all tasks in the task-set
    for check_task in taskset:  # Iterate over all tasks
        # Get response time of task: start with execution time of check_task
        response_time = _caluclate_response_time(taskset, check_task, check_task.execution_time)

        # Check schedulability of task
        if response_time > check_task.deadline:
            # Task-set is NOT schedulable
            return False

    # All tasks are schedulable -> task-set is schedulable
    return True


def rta_buttazzo(taskset):
    """Response Time Analysis according to Buttazzo.

    Check the schedulability of a task-set with response time analysis.
    Calculate the response times of all tasks. The task-set is schedulable if and only if for all
    tasks: R_i <= D_i

    Keyword arguments:
        taskset -- the task-set that should be tested
    Return value:
        True/False -- schedulability of task-set
    """
    # Check input argument: must be a TaskSet
    if not isinstance(taskset, database_interface.Taskset):  # Invalid input argument
        raise ValueError("taskset must be of type Taskset")

    # Check schedulability of all tasks in the task-set
    for check_task in taskset:  # Iterate over all tasks
        # get start value for response time calculation
        start_value = _get_start_value_buttazzo(taskset, check_task)

        # Get response time of task
        response_time = _caluclate_response_time(taskset, check_task, start_value)

        # Check schedulability of task
        if response_time > check_task.deadline:
            # Task-set is NOT schedulable
            return False

    # All tasks are schedulable -> task-set is schedulable
    return True


def _get_start_value_buttazzo(taskset, check_task):
    """Calculate the start value for response time calculation according to Buttazzo.

    The start_value for response time calculation according to Buttazzo is the sum of execution
    times of all tasks with higher or same priority as check_task.

    Args:
        taskset -- the complete task-set
        check_task -- the task, for which teh start value should be calculated
    Return:
        start_value -- the start value for response time calculation
    """
    # reset start value
    start_value = 0

    # iterate over all tasks
    for task in taskset:
        # check if priority is higher or same as check_tasks
        if task.priority <= check_task.priority:
            # add execution time of task
            start_value += task.execution_time

    # return start value
    return start_value


def _caluclate_response_time(taskset, check_task, start_value):
    """Calculate the response time of a task.

    The response time of a task i is calculated through the iterative formula:
    start:  R_0 = start_value
    iteration:  R_(k+1) = C_i + sum( R_k / T_j * C_j)
    stop:   R_(k+1) = R_k = R
    The sums over j are always over all tasks with higher or same priority than i (= hp(i)).

    Keyword arguments:
        taskset -- the task-set that should be checked
        check_task -- the task for which the response time should be calculated
        start_value -- the start value for the calculation (= response time 0)
    Return value:
        r_new -- response time of check_task
    """
    # create logger
    logger = logging.getLogger('traditional-SA.RTA._calculate_response_time')

    # check input arguments
    if not isinstance(taskset, database_interface.Taskset):
        raise ValueError("taskset must be of type Taskset")
    if not isinstance(check_task, database_interface.Task):
        raise ValueError("check_task must be of type Task")

    logger.debug("TASK %s", str(check_task.task_id))

    # Create task-set with all task of higher or same priority as check_task = hp(i)
    high_prio_set = _create_hp_set(taskset, check_task)
    logger.debug("hp-set = %s", str(high_prio_set))

    logger.debug("R0 = %s", str(start_value))

    # Check if there are tasks of higher or same priority
    if not high_prio_set:  # check_task is task with highest priority
        return start_value  # Return response time = execution time of check_task

    r_old = 0  # response time of the last iteration
    r_new = start_value  # repsonse time of the current iteration

    while r_old != r_new:  # while the response time changes with each iteration
        r_old = r_new  # save response time of last iteration

        interference = 0  # reset total interference
        # iterate over the hp-set
        for task in high_prio_set:
            # add interference of task to total interference
            interference += math.ceil(r_old / task.period) * task.execution_time

        # calculate response time of this iteration
        r_new = check_task.execution_time + interference
        logger.debug("R = %s", str(r_new))

        # check if response time is greater then deadline
        if r_new > check_task.deadline:
            # Deadline miss of check_task
            logger.debug("R > D")
            return r_new

    return r_new


def _create_hp_set(taskset, check_task):
    """Create a HP-task-set.

    This method creates a hp-task-set, i.e. a task-set with all tasks with higher or same priority
    than the check-task.

    Args:
        taskset -- the complete task-set
        check_task -- the task, for which the hp-set should be created
    Return:
        high_prio_set -- task-set with higher- and same-priority tasks
    """
    # create empty task-set
    high_prio_set = database_interface.Taskset(tasks=[])

    # iterate over all tasks
    for task in taskset:
        # check if priority is higher or same as check_tasks
        if task.priority <= check_task.priority and task is not check_task:
            # add task to hp-set
            high_prio_set.add_task(task)

    # return hp-set
    return high_prio_set
