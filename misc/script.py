#!/usr/bin/python3

"""
Example of a script that uses SimSo.
"""

import sys
from simso.core import Model
from simso.configuration import Configuration


def main(argv):
    if len(argv) == 2:
        # Configuration load from a file.
        configuration = Configuration(argv[1])
    else:
        # Manual configuration:
        configuration = Configuration()

        configuration.duration = 30

        # Add tasks:
        configuration.add_task(name="T1", uid=1, activation_date=0, jobs_activation_dates=[x * 7 for x in range(4)],
                               base_exec_cost=3, firm_deadline=7)
        configuration.add_task(name="T2", uid=2, activation_date=0, jobs_activation_dates=[x * 12 for x in range(2)],
                               base_exec_cost=3, firm_deadline=12)
        configuration.add_task(name="T3", uid=3, activation_date=0, jobs_activation_dates=[20],
                               base_exec_cost=5, firm_deadline=20)

        # Add a processor:
        configuration.add_processor(name="CPU 1", uid=1)

        # Add a scheduler:
        configuration.scheduler_info.cls = "simso.schedulers.EDF_mono"

    # Check the config before trying to run it.
    configuration.check_all()

    # Init a model from the configuration.
    model = Model(configuration)

    # Execute the simulation.
    model.run_model()

    # Print logs.
    for log in model.logs:
        print(log)


main(sys.argv)
