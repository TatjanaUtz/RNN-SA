"""Defines the scheduling policy of the Fiasco.OS scheduler.

    The algorithm is the following: a list of ready jobs is kept up-to-date using the on_activate
    and on_terminated methods. When the schedule method is called, the ready job is chosen according
    to the priorities:
    - Tasks can have priority values from 0 ... 127.
    - A priority from 0 ... 126 means, that the task is scheduled according to the fixed priority
      (FP) algorithm.
       Priority order is:   0 - highest priority
                            126 - lowest priority
    - All tasks left with priority 127 are scheduled according to the earliest deadline first (EDF)
      algorithm.

    Source:
    http://projects.laas.fr/simso/doc/write_scheduler.html
"""

import logging  # import logging for logging messages

from simso.core import Scheduler  # import scheduler class
from simso.schedulers import scheduler


# Define required task fields - must be added to task description
@scheduler("fp_edf_scheduler.py",
           required_task_fields=[{'name': 'priority', 'type': 'int', 'default': '0'}])
class fp_edf_scheduler(Scheduler):  # define fp_edf_scheduler as subclass of scheduler
    """The implementation of a scheduler.

    Subclass of abstract class Scheduler.
    The scheduling events are modeled by method calls which take as arguments the jobs and the
    processors. The init(), on_activate(), on_terminate() and schedule() methods should be redefined
    in order to interact with the simulation.
    """

    def init(self):
        """Init method.

        This method is called when the system is ready to run.
        This method is guaranteed to be called when the simulation starts, after the tasks are
        instantiated.
        The scheduler logic should be initialized here.
        """
        self.ready_list = []  # define a empty ready list

    def on_activate(self, job):
        """On_activate method.

        This method is called upon a job activation.

        Args:
            job -   the activated job
        """
        self.ready_list.append(job)  # append the job to the ready list
        job.cpu.resched()  # indirectly call the scheduler

    def on_terminated(self, job):
        """On_terminated method.

        This method is called when a job finish (termination or abortion).

        Args:
            job - the job that terminates
        """
        self.ready_list.remove(job)  # remove the job from the ready list
        job.cpu.resched()  # indirectly call the scheduler

    def schedule(self, cpu):
        """Schedule method.

        The schedule method is called by the processor (cpu) when it needs to run the scheduler.
        This method takes the scheduling decision.
        This method should not be called directly. A call to the resched method is required.

        Args:
            cpu - the processor on which the scheduler runs
        Return:
            a decision or a list of decisions, a decision is a couple (job, cpu)
        """
        # create logger
        logger = logging.getLogger('traditional-SA.fp_edf_scheduler.schedule')

        if self.ready_list:  # at least one job is ready

            # Get the lowest priority-attribute-value (i.e. the highest priority) of all ready jobs
            prio_low = min(self.ready_list, key=lambda x: x.data['priority']).data['priority']

            if 0 <= prio_low < 127:  # Lowest priority-attribute-value is less than 127
                # Schedule according to FP-algorithm
                # logging.debug("FP-algorithm!")

                # Get the job with the lowest priority-attribute-value
                # (i.e. the job with the highest priority)
                job = min(self.ready_list, key=lambda x: x.data['priority'])

            elif prio_low == 127:  # Lowest priority-attribute-value is 127
                # Schedule according to EDF-algorithm
                # logging.debug("EDF-algorithm!")

                # Get the job with the lowest deadline-attribute-value
                # (i.e. the job with the next deadline)
                job = min(self.ready_list, key=lambda x: x.absolute_deadline)

            else:  # Error: not a valid priority-attribute-value!
                logger.error("%d is not a valid priority value!", prio_low)
                return None

        else:  # no job is ready
            job = None

        return (job, cpu)
