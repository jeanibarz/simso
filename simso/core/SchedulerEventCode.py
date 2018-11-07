# coding=utf-8


class SchedulerEventCode(object):
    BEGIN_SCHEDULE = 1
    END_SCHEDULE = 2
    BEGIN_ACTIVATE = 3
    END_ACTIVATE = 4
    BEGIN_TERMINATE = 5
    END_TERMINATE = 6

    def __init__(self, cpu):
        self.event = 0
        self.cpu = cpu


class SchedulerBeginScheduleEventCode(SchedulerEventCode):
    def __init__(self, cpu):
        SchedulerEventCode.__init__(self, cpu)
        self.event = SchedulerEventCode.BEGIN_SCHEDULE


class SchedulerEndScheduleEventCode(SchedulerEventCode):
    def __init__(self, cpu):
        SchedulerEventCode.__init__(self, cpu)
        self.event = SchedulerEventCode.END_SCHEDULE


class SchedulerBeginActivateEventCode(SchedulerEventCode):
    def __init__(self, cpu):
        SchedulerEventCode.__init__(self, cpu)
        self.event = SchedulerEventCode.BEGIN_ACTIVATE


class SchedulerEndActivateEventCode(SchedulerEventCode):
    def __init__(self, cpu):
        SchedulerEventCode.__init__(self, cpu)
        self.event = SchedulerEventCode.END_ACTIVATE


class SchedulerBeginTerminateEventCode(SchedulerEventCode):
    def __init__(self, cpu):
        SchedulerEventCode.__init__(self, cpu)
        self.event = SchedulerEventCode.BEGIN_TERMINATE


class SchedulerEndTerminateEventCode(SchedulerEventCode):
    def __init__(self, cpu):
        SchedulerEventCode.__init__(self, cpu)
        self.event = SchedulerEventCode.END_TERMINATE
