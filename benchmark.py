"""Module to benchmark the execution times of tasks."""
import logging
import time

import logging_config


def benchmark_execution_times(database):
    """Benchmark to get average execution times of tasks.

    This method determines for each task the task-sets, that consist only of this task. Then all
    jobs of this task-sets and for the task are read from the database. The execution time of the
    jobs is calculated from the start- and end-date and the average value is built upon this
    execution times.

    Args:
        database -- a Database-object
    """
    logger = logging.getLogger('traditional-SA.benchmark.benchmark_execution_times')
    logger.info("Starting to benchmark execution times...")
    start_time = time.time()

    task_list = database.read_table_task(convert_to_task_dict=False)  # read table 'Task'
    c_dict = dict()  # create empty dictionary for execution times

    for task in task_list:  # iterate over all tasks
        # get all task-sets where the current task is the only task
        taskset_list = database.read_table_taskset(task_id=task[0], convert=False)

        job_attributes = []  # create empty list for the jobs

        for taskset in taskset_list:  # iterate over all task-sets
            # add all jobs of the current task and task-set
            job_attributes.extend(database.read_table_job(set_id=taskset[0], task_id=task[0]))

        # calculate execution time of each job
        job_execution_times = _calculate_executiontimes(job_attributes)

        # calculate average execution time of current task
        average_c = sum(job_execution_times) / len(job_execution_times)

        # round and add execution time to the dictionary
        c_dict[task[0]] = round(average_c)

    end_time = time.time()
    logger.info("Benchmark of execution times finished!")
    logger.info("Time elapsed: %f s", end_time - start_time)

    # write execution times to the database
    logger.info("Saving calculated execution times to database...")
    database.write_execution_time(c_dict)
    logger.info("Saving successful!")


def _calculate_executiontimes(job_attributes):
    """Calculate the execution times of the given jobs.

    This method calculates the execution times of a list of jobs with the following attributes:
        Set_ID
        Task_ID
        Job_ID
        Start_Date
        End_Date
        Exit_Value

    Args:
        job_attributes -- list with the job attributes
    Return:
        executiontimes -- list with the executiontimes
    """
    executiontimes = []  # create empty list for execution times

    # iterate over all jobs
    for job in job_attributes:
        execution_time = job[4] - job[3]  # calculate execution time = end_date - start_date

        # check if execution_time is valid
        if execution_time > 0:
            executiontimes.append(execution_time)  # append execution time to list

    return executiontimes
