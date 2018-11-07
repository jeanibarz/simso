#!/usr/bin/python
# coding=utf-8

import os
import re
from xml.dom import minidom
from simso.core.Scheduler import SchedulerInfo
from simso.core import Scheduler
from simso.core.Task import TaskInfo, AbortConditionEnum
from simso.core.Processor import ProcInfo
from .GenerateConfiguration import generate
from .parser import Parser

# Hack for Python2
if not hasattr(minidom.NamedNodeMap, '__contains__'):
    minidom.NamedNodeMap.__contains__ = minidom.NamedNodeMap.has_key


def _gcd(*numbers):
    """Return the greatest common divisor of the given integers"""
    from fractions import gcd
    return reduce(gcd, numbers)


# Least common multiple is not in standard libraries?
def _lcm(numbers):
    """Return lowest common multiple."""

    def lcm(a, b):
        return (a * b) // _gcd(a, b)

    return reduce(lcm, numbers, 1)


class Configuration(object):
    """
    The configuration class store all the details about a system. An instance
    of this class will be passed to the constructor of the
    :class:`Model <simso.core.Model.Model>` class.
    """

    def __init__(self, filename=None):
        """
        Args:
            - `filename` A file can be used to initialize the configuration.
        """
        if filename:
            parser = Parser(filename)
            self.etm = parser.etm
            self.duration = parser.duration
            self.cycles_per_ms = parser.cycles_per_ms
            self._task_info_list = parser.task_info_list
            self.task_data_fields = parser.task_data_fields
            self._proc_info_list = parser.proc_info_list
            self.proc_data_fields = parser.proc_data_fields
            self._scheduler_info = parser.scheduler_info
        else:
            self.etm = "wcet"
            self.duration = 1000
            self.cycles_per_ms = 1
            self._task_info_list = []
            self.task_data_fields = {}
            self._proc_info_list = []
            self._scheduler_info = SchedulerInfo()
        self._set_filename(filename)

    def _set_filename(self, filename):
        self._simulation_file = filename
        if filename:
            self._cur_dir = os.path.split(filename)[0]
            if not self._cur_dir:
                self._cur_dir = os.curdir
        else:
            self._cur_dir = os.curdir

    def save(self, simulation_file=None):
        """
        Save the current configuration in a file. If no file is given as
        argument, the previous file used to write or read the configuration is
        used again.
        """
        if simulation_file:
            old_dir = self._cur_dir
            self._cur_dir = os.path.split(simulation_file)[0] or '.'

            for task in self._task_info_list:
                if task.stack_file:
                    task.set_stack_file(
                        old_dir + '/' + task.stack_file, self._cur_dir)

            self._simulation_file = simulation_file

        conf_file = open(self._simulation_file, 'w')
        conf_file.write(generate(self))

    def calc_penalty_cache(self):
        for proc in self.proc_info_list:
            access_time = self.memory_access_time
            for cache in reversed(proc.caches):
                cache.penalty = access_time - cache.access_time
                access_time = cache.access_time

            proc.penalty = access_time

    def check_all(self):
        """
        Check the correctness of the configuration (without simulating it).
        """
        self.check_general()
        self.check_scheduler()
        self.check_processors()
        self.check_tasks()

    def check_general(self):
        assert self.duration >= 0, \
            "Simulation duration must be a positive number."
        assert self.cycles_per_ms >= 0, \
            "Cycles / ms must be a positive number."

    def check_scheduler(self):
        cls = self._scheduler_info.get_cls()
        assert cls is not None, \
            "A scheduler is needed."
        assert issubclass(cls, Scheduler), \
            "Must inherits from Scheduler."

    def check_processors(self):
        # At least one processor:
        assert len(self._proc_info_list) > 0, \
            "At least one processor is needed."

        for index, proc in enumerate(self._proc_info_list):
            # Nom correct :
            assert re.match('^[a-zA-Z][a-zA-Z0-9 _-]*$', proc.name), \
                "A processor name must begins with a letter and must not " \
                "contains any special character."
            # Id unique :
            assert proc.uid not in [
                x.uid for x in self._proc_info_list[index + 1:]], \
                "Processors' identifiers must be uniques."

    def check_tasks(self):
        assert len(self._task_info_list) > 0, "At least one task is needed."
        for index, task_info in enumerate(self._task_info_list):
            # Id unique :
            assert task_info.uid not in [
                x.uid for x in self._task_info_list[index + 1:]], \
                "Tasks' identifiers must be uniques."
            # Nom correct :
            assert re.match('^[a-zA-Z][a-zA-Z0-9 _-]*$', task_info.name), "A task " \
                                                                          "name must begins with a letter and must not contains any " \
                                                                          "special character."

            #  Activation date >= 0:
            assert task_info.activation_date >= 0, \
                "Activation date must be positive."

            # All jobs activation dates >= 0
            assert all(x >= 0 for x in task_info.jobs_activation_dates), "Jobs activation dates must be positives"

            #  Deadline >= 0:
            assert task_info.firm_deadline >= 0, "Tasks' deadlines must be positives."

    def get_hyperperiod(self):
        """
        Compute and return the hyperperiod of the tasks.
        """
        return _lcm([x.period for x in self.task_info_list])

    @property
    def duration_ms(self):
        return self.duration / self.cycles_per_ms

    @property
    def simulation_file(self):
        return self._simulation_file

    @property
    def cur_dir(self):
        return self._cur_dir

    @property
    def caches_list(self):
        return self._caches_list

    @property
    def task_info_list(self):
        """
        List of tasks (TaskInfo objects).
        """
        return self._task_info_list

    @property
    def proc_info_list(self):
        """
        List of processors (ProcInfo objects).
        """
        return self._proc_info_list

    @property
    def scheduler_info(self):
        """
        SchedulerInfo object.
        """
        return self._scheduler_info

    def add_task(self, name, uid, activation_date, jobs_activation_dates, base_exec_cost, firm_deadline,
                 base_utility_value=1, soft_deadline=0, abort_condition=AbortConditionEnum.ON_SOFT_DEADLINE):
        """
        Helper method to create a TaskInfo and add it to the list of tasks.
        """
        task_info = TaskInfo(name=name, uid=uid, activation_date=activation_date,
                             jobs_activation_dates=jobs_activation_dates,
                             base_utility_value=base_utility_value, base_exec_cost=base_exec_cost,
                             firm_deadline=firm_deadline, soft_deadline=soft_deadline, abort_condition=abort_condition)
        self.task_info_list.append(task_info)
        return task_info

    def add_processor(self, name, uid, exec_speed=1.0):
        """
        Helper method to create a ProcInfo and add it to the list of
        processors.
        """
        proc = ProcInfo(name=name, uid=uid, exec_speed=exec_speed)
        self.proc_info_list.append(proc)
        return proc
