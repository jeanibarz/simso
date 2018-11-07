# coding=utf-8

from collections import deque
from SimPy.Simulation import Process, Monitor, hold, passivate
from simso.core.Job import Job
from simso.core.Timer import Timer
from enum import Enum
from typing import List


class AbortConditionEnum(Enum):
    ON_FIRM_DEADLINE = "ON_FIRM_DEADLINE",
    ON_SOFT_DEADLINE = "ON_SOFT_DEADLINE"


class TaskInfo(object):
    """
    TaskInfo is mainly a container class grouping the data that characterize
    a Task. A list of TaskInfo objects are passed to the Model so that
    :class:`Task` instances can be created.
    """

    def __init__(self, name: str, uid, activation_date: int, jobs_activation_dates: List[int], base_utility_value: int,
                 base_exec_cost: int, firm_deadline: int, soft_deadline: int, abort_condition: AbortConditionEnum):
        """
        TODO: document

        :param name:
        :param uid:
        :param activation_date:
        :param jobs_activation_dates:
        :param base_utility_value:
        :param base_exec_cost:
        :param firm_deadline:
        :param soft_deadline:
        :param abort_condition:
        """
        self.name = name
        self.uid = uid  # object unique identifier
        self.activation_date = activation_date
        self.jobs_activation_dates = jobs_activation_dates
        self.base_utility_value = base_utility_value
        self.base_exec_cost = base_exec_cost
        self.firm_deadline = firm_deadline
        self.soft_deadline = soft_deadline
        self.abort_condition = abort_condition

class SoftTask(Process):
    next_uid = 0

    @classmethod
    def get_new_uid(cls):
        uid = cls.next_uid
        cls.next_uid += 1
        return uid

    def __init__(self, sim, task_info: TaskInfo):
        """
        Args:

        - `sim`: :class:`Model <simso.core.Model>` instance.
        - `task_info`: A :class:`TaskInfo` representing the Task.

        :type sim: Model
        :type task_info: TaskInfo
        """
        Process.__init__(self, name=task_info.name, sim=sim)
        self._cpu = self.sim.processors_list[0]  # run jobs on first cpu
        self._uid = SoftTask.get_new_uid()
        self.name = task_info.name
        self._task_info = task_info
        self._monitor = Monitor(name="Monitor" + self.name + "_states",
                                sim=sim)
        self._sim = sim  # should be removed ?
        self._etm = sim.etm  # should be removed ?
        self._job_count = 0  # ???
        self._jobs = []  # list of all created jobs
        self.job = None  # current active job, should be refactored so that multiple jobs can be active at the same time

    def __lt__(self, other):
        return self.uid < other.uid  # no way !

    @property
    def cpu(self):
        return self._cpu

    @property
    def firm_deadline(self):
        """
        Deadline in milliseconds.
        """
        return self._task_info.firm_deadline

    @property
    def soft_deadline(self):
        return self._task_info.soft_deadline

    @property
    def base_exec_cost(self):
        return self._task_info.base_exec_cost

    @property
    def uid(self):
        """
        Identifier of the task.
        """
        return self._uid

    @property
    def monitor(self):
        """
        The monitor for this Task. Similar to a log mechanism (see Monitor in
        SimPy doc).
        """
        return self._monitor

    @property
    def jobs(self):
        """
        List of the jobs.
        """
        return self._jobs

    # TODO: to be modified, a task has not to start automatically a new jobs : they are activated following the activation dates only
    def end_job(self, job):
        self._last_cpu = self.cpu

    # TODO: to be removed ?
    def _job_killer(self, job):
        job.abort()

    # TODO: to be removed ?
    def create_job(self, ref_time, cpu, pred=None):
        """
        Create a new job from this task. This should probably not be used
        directly by a scheduler.
        """
        self._job_count += 1
        job = Job(self, name="{}_{}".format(self.name, self._job_count), monitor=self._monitor, etm=self._etm, sim=self.sim)

        self.job = job
        self.sim.activate(job, job.activate_job(cpu=cpu))
        self._jobs.append(job)

        if self._task_info.abort_condition is AbortConditionEnum.ON_SOFT_DEADLINE:
            abort_date = self.soft_deadline
        elif self._task_info.abort_condition is AbortConditionEnum.ON_FIRM_DEADLINE:
            abort_date = self.firm_deadline
        timer_job_abort = Timer(self.sim, SoftTask._job_killer,
                                (self, job), abort_date)
        timer_job_abort.start()


class SporadicTask(SoftTask):
    """
    (Sporadic, SoftTask) process. Inherits from :class:`SoftTask`. The jobs are
    created using a list of activation dates.
    """

    def __init__(self, sim, task_info: TaskInfo):
        super().__init__(sim, task_info)

    def execute(self, cpu):
        for ai in self.jobs_activation_dates:
            # TODO: use Timers instead of yield to create the jobs
            yield hold, self, int(ai * self._sim.cycles_per_ms) - self._sim.now()
            self.create_job(ref_time=ai, cpu=cpu)

    @property
    def jobs_activation_dates(self):
        return self._task_info.jobs_activation_dates
