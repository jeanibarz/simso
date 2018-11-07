# coding=utf-8

from collections import deque
from SimPy.Simulation import Process, Monitor, hold, waituntil
from simso.core.ProcEventCode import ProcRunEventCode, ProcIdleEventCode, \
    ProcOverheadEventCode, ProcCxtSaveEvent, ProcCxtLoadEvent


RESCHED = 1
ACTIVATE = 2
TERMINATE = 3
TIMER = 4
PREEMPT = 5
SPEED = 6


class ProcInfo(object):
    def __init__(self, uid, name, exec_speed=1.0):
        self.uid = uid
        self.name = name
        self.exec_speed = exec_speed

    def add_cache(self, cache):
        self.caches.append(cache)


class Processor(Process):
    """
    A processor is responsible of deciding whether the simulated processor
    should execute a job or execute the scheduler. There is one instance of
    Processor per simulated processor. Those are responsible to call the
    scheduler methods.

    When a scheduler needs to take a scheduling decision, it must invoke the
    :meth:`resched` method. This is typically done in the :meth:`on_activate
    <simso.core.Scheduler.Scheduler.on_activate>`, :meth:`on_terminated
    <simso.core.Scheduler.Scheduler.on_terminated>` or in a :class:`timer
    <simso.core.Timer.Timer>` handler.
    """
    next_uid = 0

    @classmethod
    def get_new_uid(cls):
        uid = cls.next_uid
        cls.next_uid += 1
        return uid

    @classmethod
    def init(cls):
        cls.next_uid = 0

    def __init__(self, model, proc_info):
        Process.__init__(self, name=proc_info.name, sim=model)
        self._proc_info = proc_info
        self._model = model
        self._uid = Processor.get_new_uid()
        self._running = None
        self.was_running = None
        self._evts = deque([])
        self.sched = model.scheduler
        self.monitor = Monitor(name="Monitor" + proc_info.name, sim=model)
        self.timer_monitor = Monitor(name="Monitor Timer" + proc_info.name,
                                     sim=model)

    def resched(self):
        """
        Add a resched event to the list of events to handle.
        """
        self._evts.append((RESCHED,))

    def activate(self, job):
        self._evts.append((ACTIVATE, job))

    def terminate(self, job):
        self._evts.append((TERMINATE, job))
        self._running = None

    def preempt(self, job=None):
        self._evts = deque([e for e in self._evts if e[0] != PREEMPT])
        self._evts.append((PREEMPT,))
        self._running = job

    def timer(self, timer):
        self._evts.append((TIMER, timer))

    def set_speed(self, speed):
        assert speed >= 0, "Speed must be positive."
        self._evts.append((SPEED, speed))

    @property
    def speed(self):
        return self._proc_info.exec_speed

    def is_running(self):
        """
        Return True if a job is currently running on that processor.
        """
        return self._running is not None

    def set_caches(self, caches):
        self._caches = caches
        for cache in caches:
            cache.shared_with.append(self)

    def get_caches(self):
        return self._caches

    caches = property(get_caches, set_caches)

    @property
    def penalty_memaccess(self):
        return self._penalty

    @property
    def cs_overhead(self):
        return self._cs_overhead

    @property
    def cl_overhead(self):
        return self._cl_overhead

    @property
    def internal_id(self):
        """A unique, internal, id."""
        return self._uid

    @property
    def running(self):
        """
        The job currently running on that processor. None if no job is
        currently running on the processor.
        """
        return self._running

    def run(self):
        while True:
            if not self._evts:
                job = self._running
                if job:
                    job.interruptReset()
                    self.sim.reactivate(job)
                    self.monitor.observe(ProcRunEventCode(job))
                else:
                    self.monitor.observe(ProcIdleEventCode())

                # Wait event.
                yield waituntil, self, lambda: self._evts
                if job:
                    self.interrupt(job)

            evt = self._evts.popleft()
            if evt[0] == RESCHED:
                if any(x[0] != RESCHED for x in self._evts):
                    self._evts.append(evt)
                    continue

            if evt[0] == ACTIVATE:
                self.sched.on_activate(evt[1])
                self.monitor.observe(ProcOverheadEventCode("JobActivation"))
            elif evt[0] == TERMINATE:
                self.sched.on_terminated(evt[1])
                self.monitor.observe(ProcOverheadEventCode("JobTermination"))
            elif evt[0] == TIMER:
                self.timer_monitor.observe(None)
                if evt[1].overhead > 0:
                    print(self.sim.now(), "hold", evt[1].overhead)
                    yield hold, self, evt[1].overhead
                evt[1].call_handler()
            elif evt[0] == SPEED:
                self._exec_speed = evt[1]
            elif evt[0] == RESCHED:
                self.monitor.observe(ProcOverheadEventCode("Scheduling"))
                self.sched.monitor_begin_schedule(self)
                yield waituntil, self, self.sched.get_lock
                decisions = self.sched.schedule(self)
                if type(decisions) is not list:
                    decisions = [decisions]
                decisions = [d for d in decisions if d is not None]

                for job, cpu in decisions:
                    # If there is nothing to change, simply ignore:
                    if cpu.running == job:
                        continue

                    # If trying to execute a terminated job, warn and ignore:
                    if job is not None and not job.is_active():
                        print("Can't schedule a terminated job! ({})"
                              .format(job.name))
                        continue

                    # if the job was running somewhere else, stop it.
                    if job and job.cpu.running == job:
                        job.cpu.preempt()

                    # Send that job to processor cpu.
                    cpu.preempt(job)

                # Forbid to run a job simultaneously on 2 or more processors.
                running_tasks = [
                    cpu.running.name
                    for cpu in self._model.processors_list if cpu.running]
                assert len(set(running_tasks)) == len(running_tasks), \
                    "Try to run a job on 2 processors simultaneously!"

                self.sched.release_lock()
                self.sched.monitor_end_schedule(self)
