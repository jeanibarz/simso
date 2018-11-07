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
    def set_uid(cls):
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
        self._uid = SoftTask.next_uid
        self.name = task_info.name  # should be removed ?
        self._task_info = task_info  # should be removed ?
        self._monitor = Monitor(name="Monitor" + self.name + "_states",
                                sim=sim)
        self._activations_fifo = deque([])
        self._sim = sim  # should be removed ?
        self.cpu = None  # should be removed ?
        self._etm = sim.etm  # should be removed ?
        self._job_count = 0  # ???
        self._last_cpu = None  # should be removed ?
        self._cpi_alone = {}  # ???
        self._jobs = []  # list of all created jobs
        self.job = None  # current active job, should be refactored so that multiple jobs can be active at the same time

    def __lt__(self, other):
        return self.identifier < other.identifier  # no way !

    @property
    def data(self):
        """
        Extra data to characterize the task. Only used by the scheduler.
        """
        return self._task_info.data

    @property
    def deadline(self):
        """
        Deadline in milliseconds.
        """
        return self._task_info.firm_deadline

    @property
    def period(self):
        """
        Period of the task.
        """
        return self._task_info.period

    @property
    def identifier(self):
        """
        Identifier of the task.
        """
        return self._task_info.identifier

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

        # Remove the job
        if len(self._activations_fifo) > 0:
            self._activations_fifo.popleft()
        # Activate the next job
        if len(self._activations_fifo) > 0:
            self.job = self._activations_fifo[0]
            self.sim.activate(self.job,
                              self.job.activate_job())  # TODO: refactor this to remove simso.core.Model dependancy

    # TODO: to be removed ?
    def _job_killer(self, job):
        if job.end_date is None and job.computation_time < job.wcet:
            if self._task_info.abort_on_miss:
                self.cancel(job)
                job.abort()

    # TODO: to be removed ?
    def create_job(self, ref_time, pred=None):
        """
        Create a new job from this task. This should probably not be used
        directly by a scheduler.
        """
        self._job_count += 1
        job = Job(self, name="{}_{}".format(self.name, self._job_count), monitor=self._monitor, etm=self._etm, sim=self.sim)

        if len(self._activations_fifo) == 0:
            self.job = job
            self.sim.activate(job, job.activate_job())
        self._activations_fifo.append(job)
        self._jobs.append(job)

        timer_deadline = Timer(self.sim, SoftTask._job_killer,
                               (self, job), self.deadline)
        timer_deadline.start()

    def _init(self):
        if self.cpu is None:
            self.cpu = self._sim.processors[0]


class SporadicTask(SoftTask):
    """
    (Sporadic, SoftTask) process. Inherits from :class:`GenericTask`. The jobs are
    created using a list of activation dates.
    """

    def execute(self):
        self._init()
        for ndate in self.jobs_activation_dates:
            yield hold, self, int(ndate * self._sim.cycles_per_ms) \
                  - self._sim.now()
            self.create_job(ref_time=ndate)
        # TODO: replace previous for loop by this next one
        for ai in self.jobs_activation_dates:
            self.create_job(ref_time=ai)

    @property
    def jobs_activation_dates(self):
        return self._task_info.jobs_activation_dates
