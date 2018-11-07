from simso.core.etm.AbstractExecutionTimeModel \
    import AbstractExecutionTimeModel


class WCET(AbstractExecutionTimeModel):
    def __init__(self, sim, _):
        self.sim = sim  # should be removed ?
        self.executed = {}
        self.on_execute_date = {}

    def init(self):
        pass

    # TODO: add (current_time, cpu_speed) parameter to remove resp. simso.core.Model dependancy and simso.core.Processor dependancy
    def update_executed(self, job):
        if job in self.on_execute_date:
            self.executed[job] += (self.sim.now() - self.on_execute_date[job]
                                   ) * job.cpu.speed

            del self.on_execute_date[job]

    def on_activate(self, job):
        self.executed[job] = 0

    # TODO: add current_time parameter to remove simso.core.Model dependancy
    def on_execute(self, job):
        self.on_execute_date[job] = self.sim.now()

    def on_preempted(self, job):
        self.update_executed(job)

    def on_terminated(self, job):
        self.update_executed(job)

    def on_abort(self, job):
        self.update_executed(job)

    # TODO: add (current_time, cpu_speed) parameter to remove resp. simso.core.Model dependancy and simso.core.Processor dependancy
    def get_executed(self, job):
        if job in self.on_execute_date:
            c = (self.sim.now() - self.on_execute_date[job]) * job.cpu.speed
        else:
            c = 0
        return self.executed[job] + c

    # TODO: add cycles_per_ms parameter to remove simso.core.Model dependancy
    def get_ret(self, job):
        # WARNING : possible side effects due to truncatures here !
        wcet_cycles = int(job.base_exec_cost * self.sim.cycles_per_ms)
        return int(wcet_cycles - self.get_executed(job))

    def update(self):
        for job in list(self.on_execute_date.keys()):
            self.update_executed(job)
