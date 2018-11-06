from simso.generator.TaskGenerator import gen_kato_aperiodic_tasks_set


def test_generating_set():
    task_info_list = gen_kato_aperiodic_tasks_set(total_utilization=1.0,
                                                  total_duration=1000,
                                                  periods_dist={'dist_type': 'uniform', 'min': 100, 'max': 1000, 'round_to_integer': True},
                                                  max_utilization_relative_error=0.05,
                                                  values_dist={'dist_type': 'uniform', 'min': 100, 'max': 1000, 'round_to_integer': False},
                                                  firm_tol_dist={'dist_type': 'uniform', 'min': 100, 'max': 1000, 'round_to_integer': True},
                                                  soft_tol_dist={'dist_type': 'uniform', 'min': 100, 'max': 1000, 'round_to_integer': True})
    assert len(task_info_list) > 0

