"""
Tools for generating task sets.
"""

import copy
import math
import random
from collections import namedtuple
from enum import Enum
from typing import List

import numpy as np

from simso.core.Task import TaskInfo


class GeneratorEnum(Enum):
    UNDEFINED = 0
    STAFFORD_RAND_FIXED_SUM = 1
    UUNI_FAST_DISCARD = 2
    KATO = 3
    RIPOLL = 4


class DistTypeEnum(Enum):
    UNIFORM = 0
    LOGUNIFORM = 1
    DISCRETE = 2


def UUniFastDiscard(n, u) -> list:
    """
        The UUniFast algorithm was proposed by Bini for generating task
        utilizations on uniprocessor architectures.

        The UUniFast-Discard algorithm extends it to multiprocessor by
        discarding task sets containing any utilization that exceeds 1.

        This algorithm is easy and widely used. However, it suffers from very
        long computation times when n is close to u. Stafford's algorithm is
        faster.

        Args:
            - `n`: The number of tasks in a task set.
            - `u`: Total utilization of the task set.

        Returns a list of task utilizations.
    """

    while True:
        # Classic UUniFast algorithm:
        utilizations = []
        remainingU = u
        for i in range(1, n):
            r = random.random() ** (1.0 / (n - i))
            utilizations.append(remainingU * (1 - r))
            remainingU *= r
        utilizations.append(remainingU)

        # If no task utilization exceeds 1:
        if not [ut for ut in utilizations if ut > 1]:
            return utilizations


def StaffordRandFixedSum(n, u) -> list:
    """
    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
    EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The views and conclusions contained in the software and documentation are
    those of the authors and should not be interpreted as representing official
    policies, either expressed or implied, of Paul Emberson, Roger Stafford or
    Robert Davis.

    Includes Python implementation of Roger Stafford's randfixedsum implementation
    http://www.mathworks.com/matlabcentral/fileexchange/9700
    Adapted specifically for the purpose of taskset generation with fixed
    total utilisation value

    Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
    any questions regarding this software.

    -----

    Stafford's RandFixedSum algorithm implementated in Python.

    Based on the Python implementation given by Paul Emberson, Roger Stafford,
    and Robert Davis. Available under the Simplified BSD License.

    Args:
        - `n`: The number of tasks in a task set.
        - `u`: Total utilization of the task set.
        - `nsets`: Number of sets to generate.
    """
    if n < u:
        return None

    # deal with n=1 case
    if n == 1:
        return [u]

    raise NotImplementedError()

    # k = min(int(u), n - 1)
    # s = u
    # s1 = s - np.arange(k, k - n, -1.)
    # s2 = np.arange(k + n, k, -1.) - s
    #
    # tiny = np.finfo(float).tiny
    # huge = np.finfo(float).max
    #
    # w = np.zeros((n, n + 1))
    # w[0, 1] = huge
    # t = np.zeros((n - 1, n))
    #
    # for i in np.arange(2, n + 1):
    #     tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
    #     tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
    #     w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
    #     tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
    #     tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
    #     t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
    #                                 (1 - tmp1 / tmp3) * (np.logical_not(tmp4))
    #
    # x = np.zeros(n)
    # rt = np.random.uniform(size=n - 1)  # rand simplex type
    # rs = np.random.uniform(size=n - 1)  # rand position in simplex
    # s = np.repeat(s)
    # j = np.repeat(k + 1)
    # sm = np.repeat(0)
    # pr = np.repeat(1)
    #
    # for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
    #     # decide which direction to move in this dimension (1 or 0):
    #     e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
    #     sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
    #     sm = sm + (1.0 - sx) * pr * s / (i + 1)
    #     pr = sx * pr
    #     x[(n - i) - 1, ...] = sm + pr * e
    #     s = s - e
    #     j = j - e  # change transition table column if required
    #
    # x[n - 1, ...] = sm + pr * s
    #
    # # iterated in fixed dimension order but needs to be randomised
    # # permute x row order within each column
    # for i in range(0, nsets):
    #     x[..., i] = x[np.random.permutation(n), i]
    #
    # return x.T.tolist()


def Ripoll(compute, deadline, period, target_util) -> List[tuple]:
    """
    Ripoll et al. tasksets generator.

    Args:
        - `compute`: Maximum computation time of a task.
        - `deadline`: Maximum slack time.
        - `period`: Maximum delay after the deadline.
        - `target_util`: Total utilization to reach.

    Returns a list of tuples (c, d, p) where :
        - `c` is the task's job cost,
        - `d` is the task's deadline,
        - `p` is the task's period
    """
    task_set = []
    total_util = 0.0
    while total_util < target_util:
        c = random.randint(1, compute)
        d = c + random.randint(0, deadline)
        p = d + random.randint(0, period)
        task_set.append((c, d, p))
        total_util += float(c) / p
    return task_set


def Kato(umin, umax, target_util) -> list:
    """
    Kato et al. tasksets generator.

    A task set Γ is generated as follows. A new periodic task is appended
    to Γ as long as U(Γ) ≤ Utot is satisfied. For each task τi, its
    utilization Ui is computed based on a uniform distribution within the
    range of [Umin, Umax]. Only the utilization of the task generated at the
    very end is adjusted so that U(Γ) becomes equal to Utot (thus the Umin
    constraint might not be satisfied for this task).

    Args:
        - `umin`: Minimum task utilization.
        - `umax`: Maximum task utilization.
        - `target_util`:

    Returns a list of tasks utilizations
    """
    task_set = []
    total_util = 0.0
    while total_util < target_util:
        u = random.uniform(umin, umax)
        if u + total_util > target_util:
            u = target_util - total_util
            if u < umin:
                raise RuntimeWarning(
                    'Minimal task utilization constraint umin>=%d unsatisfied for one task'.format({umin}))
        total_util += u
        task_set.append(u)
    return task_set


def next_arrival_poisson(period):
    return -math.log(1.0 - random.random()) * period


def gen_poisson_arrivals(period, min_, max_, round_to_int=False) -> list:
    def trunc(x, p):
        return int(x * 10 ** p) / float(10 ** p)

    dates = []
    n = min_ - period
    while True:
        n += next_arrival_poisson(period) + period
        if round_to_int:
            n = int(round(n))
        else:
            n = trunc(n, 6)
        if n > max_:
            break
        dates.append(n)
    return dates


def gen_random_values(dist_type: DistTypeEnum, n, low=0, high=1, round_to_int=False, choices=[]):
    if dist_type is DistTypeEnum.UNIFORM:
        periods = np.random.uniform(low=low, high=high, size=n)
    elif dist_type is DistTypeEnum.LOGUNIFORM:
        periods = np.exp(np.random.uniform(low=np.log(low), high=np.log(high),
                                           size=n))
    elif dist_type is DistTypeEnum.DISCRETE:
        assert len(choices) >= 1
        try:
            periods = np.random.choice(choices, size=n)
        except AttributeError:
            # Numpy < 1.7:
            periods = [choices[i] for i in np.random.randint(low=0, high=len(choices), size=n).tolist()]
    if round_to_int:
        return np.rint(periods).tolist()
    else:
        return periods.tolist()


def gen_tasks_costs(utilizations, periods) -> list:
    """
    Take a list of task utilization and a list of task period and
    return a list of couples (c, p). The computation times are truncated
    at a precision of 10^-10 to avoid floating point precision errors.

    Args:
        - `utilization`: The list of task utilization. For example::

            [0.3, 0.4, 0.8, 0.1, 0.9, 0.5]
        - `periods`: The list of task period. For examples::

            [100, 50, 1000, 200, 500, 10]

    Returns:
        For the above example, it returns::

            [(30.0, 100), (20.0, 50), (800.0, 1000), (20.0, 200), (450.0, 500), (5.0, 10)]
    """

    def trunc(x, p):
        return int(x * 10 ** p) / float(10 ** p)

    return [(trunc(ui * pi, 6), trunc(pi, 6)) for ui, pi in zip(utilizations, periods)]


def gen_kato_aperiodic_tasks_set(total_utilization: float, total_duration, periods_dist, max_utilization_relative_error=0.05,
                                 values_dist=None, firm_tol_dist=None, soft_tol_dist=None, ) -> List[TaskInfo]:
    """
    Args :
    :param total_utilization:
    :param total_duration:
    :param periods_dist: period sampling type
    :param max_utilization_relative_error:
    :param values_dist: dictionary {'dist_type','min','max'}
    :param firm_tol_dist: dictionary {'dist_type','min','max'}
    :param soft_tol_dist: dictionary {'dist_type','min','max'}
    :return: a list of TaskInfo objects
    """
    tasks_caracs = namedtuple('tasks_caracs',
                              ['tasks_nbr', 'utilizations', 'periods', 'values', 'jobs_nbr', 'jobs_execution_cost',
                               'firm_tol', 'soft_tol'])
    # Check arguments
    if not total_utilization >= 0:
        raise ValueError('total_utilization must be non negative')

    # Defines tasks periods (mean inter-arrival time)
    generic_map_dist_types = {'unif': DistTypeEnum.UNIFORM, 'lunif': DistTypeEnum.LOGUNIFORM,
                              'discrete': DistTypeEnum.DISCRETE}
    try:
        periods_dist_type = generic_map_dist_types[periods_dist['dist_type']]
    except KeyError:
        raise ValueError('period distribution type is invalid')
    tasks_caracs.periods = gen_random_values(dist_type=periods_dist_type,
                                             n=tasks_caracs.tasks_nbr, low=periods_dist['min'],
                                             high=periods_dist['max'],
                                             round_to_int=periods_dist['round_to_integer'])

    # Gets an integer number of jobs that fits in the simulation
    tasks_caracs.jobs_nbr = [math.floor(total_duration / period) for period in tasks_caracs.periods]

    valid_utilization = False
    for utilization_nbr_retry in range(100):
        # Use Kato algorithm to define each task's utilization value
        max_jobs_nbr = math.floor(total_duration / periods_dist['min'])
        min_utilization = 3 * max_jobs_nbr / total_duration
        max_utilization = 0.5
        if not min_utilization >= 0:
            raise ValueError('min_utilization must be positive')
        if not (max_utilization >= 0 and max_utilization <= 1):
            raise ValueError('max_utilization must be in [0;1]')
        if not min_utilization <= max_utilization:
            raise ValueError('min_utilization must be less (or equal) then max_utilization')

        valid_kato = False
        for kato_nbr_retry in range(0, 100):
            try:
                tasks_caracs.utilizations = Kato(umin=min_utilization,
                                                 umax=max_utilization,
                                                 target_util=total_utilization, error_tol=5.0 / 100)
                tasks_caracs.tasks_nbr = len(tasks_caracs.utilizations)
                if tasks_caracs.tasks_nbr > 0:
                    valid_kato = True
                    break
            except RuntimeWarning:
                continue
        if not valid_kato:
            raise RuntimeError('Kato generation failed {} consecutive times !'.format(kato_nbr_retry))

        assert len(tasks_caracs.utilizations) == tasks_caracs.tasks_nbr

        # Generate task's jobs costs (a constant execution speed of 1.0 is assumed)
        tasks_caracs.jobs_execution_cost = [ui * (total_duration / ni) for ui, ni in
                                            zip(tasks_caracs.utilizations, tasks_caracs.jobs_nbr)]

        # Adjust task's jobs costs
        error = 0
        tasks_set = sorted(
            list(zip(tasks_caracs.jobs_nbr, tasks_caracs.jobs_execution_cost, range(0, tasks_caracs.tasks_nbr))),
            reverse=True)
        for ni, ci, i in tasks_set:
            new_ci_upper = math.ceil(ci)
            assert new_ci_upper > 0

            l_new_ci = []
            for j in range(0, min(new_ci_upper, 3)):
                l_new_ci.append(new_ci_upper - j)
            for j in range(1, 3):
                l_new_ci.append(new_ci_upper + j)

            # select the cost that will minimize the new aggregated error
            l_abs_new_errors = list(map(lambda _new_ci: abs(error + (_new_ci - ci) * ni), l_new_ci))
            best_ci_index = np.argmin(l_abs_new_errors)
            new_ci = l_new_ci[best_ci_index]
            assert new_ci > 0
            # prev_error = error
            # new_error = error + (new_ci - ci) * ni
            error += (new_ci - ci) * ni
            tasks_caracs.jobs_execution_cost[i] = new_ci
        del tasks_set
        del error

        total_execution_cost = sum([ni * ci for ni, ci in zip(tasks_caracs.jobs_nbr, tasks_caracs.jobs_execution_cost)])
        obtained_total_utilization = total_execution_cost / total_duration
        relative_utilization_error = (obtained_total_utilization - total_utilization) / total_utilization
        if abs(relative_utilization_error) <= max_utilization_relative_error:
            valid_utilization = True
            break
    if not valid_utilization:
        raise RuntimeError('Failed to satisfy max_utilization_relative_error after {} consecutive times !'.format(
            utilization_nbr_retry))

    # Defines tasks firm tolerances (or base laxity : relative deadline - ci)
    try:
        firm_tol_dist_type = generic_map_dist_types[firm_tol_dist['dist_type']]
    except KeyError:
        raise ValueError('firm_tol distribution type is invalid')
    tasks_caracs.firm_tol = gen_random_values(dist_type=firm_tol_dist_type,
                                              n=tasks_caracs.tasks_nbr, low=firm_tol_dist['min'],
                                              high=firm_tol_dist['max'],
                                              round_to_int=firm_tol_dist['round_to_integer'])

    # Defines tasks soft tolerances
    try:
        soft_tol_dist_type = generic_map_dist_types[soft_tol_dist['dist_type']]
    except KeyError:
        raise ValueError('firm_tol distribution type is invalid')
    tasks_caracs.firm_tol = gen_random_values(dist_type=soft_tol_dist_type,
                                              n=tasks_caracs.tasks_nbr, low=soft_tol_dist['min'],
                                              high=soft_tol_dist['max'],
                                              round_to_int=soft_tol_dist['round_to_integer'])

    # Defines tasks values
    if values_dist['dist_type'] == 'wcet_prop':
        tasks_caracs.values = [ci for ci in tasks_caracs.jobs_execution_cost]
    elif values_dist['dist_type'] == 'inv_wcet_prop':
        tasks_caracs.values = [1.0 / ci for ci in tasks_caracs.jobs_execution_cost]
    else:
        try:
            values_dist_type = generic_map_dist_types[values_dist['dist_type']]
        except KeyError:
            raise ValueError('firm_tol distribution type is invalid')
        tasks_caracs.values = gen_random_values(dist_type=values_dist_type,
                                                n=tasks_caracs.tasks_nbr, low=values_dist['min'],
                                                high=values_dist['max'],
                                                round_to_int=values_dist['round_to_integer'])

    # Defines tasks base data
    base_data = {'rt_constraint': 'SOFT_RT', 'importance_value': 1, 'deadline_tolerance': 0}

    # Defines tasks set
    task_info_list = []
    for i in range(tasks_caracs.tasks_nbr):
        task_data = copy.deepcopy(base_data)
        ci = tasks_caracs.jobs_execution_cost[i]
        vi = tasks_caracs.values[i]
        ni = tasks_caracs.jobs_nbr[i]
        di = ci + tasks_caracs.firm_tol[i]

        task_data['deadline_tolerance'] = tasks_caracs.soft_tol[i]
        task_data['importance_value'] = vi

        list_activation_dates = gen_random_values(dist_type=DistTypeEnum.UNIFORM, n=ni, low=0,
                                                  high=total_duration - math.ceil(ci),
                                                  round_to_int=True)
        task_info_list.append(TaskInfo(name="Task " + str(i), identifier=i, task_type="Sporadic",
                                       abort_on_miss=False, period=None,
                                       activation_date=0, n_instr=0, mix=0,
                                       stack_file=('', ''), wcet=ci, acet=0, et_stddev=0,
                                       deadline=di, base_cpi=0.0, followed_by=None,
                                       list_activation_dates=list_activation_dates,
                                       preemption_cost=0, data=task_data))

    return task_info_list
