# coding=utf-8


class ProcEventCode(object):
    RUN = 1
    IDLE = 2
    OVERHEAD = 3

    def __init__(self, event=0, args=None):
        self.event = event
        self.args = args


class ProcRunEventCode(ProcEventCode):
    def __init__(self, job):
        ProcEventCode.__init__(self, ProcEventCode.RUN, job)


class ProcIdleEventCode(ProcEventCode):
    def __init__(self):
        ProcEventCode.__init__(self, ProcEventCode.IDLE)


class ProcOverheadEventCode(ProcEventCode):
    def __init__(self, type_overhead):
        ProcEventCode.__init__(self, ProcEventCode.OVERHEAD, type_overhead)


class ProcCxtSaveEvent(ProcOverheadEventCode):
    def __init__(self, terminated=False):
        ProcOverheadEventCode.__init__(self, "CS")
        self.terminated = terminated


class ProcCxtLoadEvent(ProcOverheadEventCode):
    def __init__(self, terminated=False):
        ProcOverheadEventCode.__init__(self, "CL")
        self.terminated = terminated
