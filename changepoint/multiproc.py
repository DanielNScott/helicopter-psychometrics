"""Parallel implementations of task generation and subject simulation."""

from copy import deepcopy
import multiprocessing as mp
import numpy as np

from changepoint.subjects import Subject, simulate_subject, DEFAULT_PARAMS_SUBJ
from changepoint.tasks import simulate_cpt


# --- Parallel task generation ---

def _worker_task_job(args):
    """Worker function for parallel task generation."""
    job_idx, s, t, task_params, blocks, seed = args
    np.random.seed(seed)
    task = simulate_cpt(task_params, blocks=blocks)
    return {'idx': job_idx, 's': s, 't': t, 'task': task}


def generate_tasks_parallel(n_subjects, n_tasks_per_subject, task_params=None, blocks=None, n_processes=None):
    """
    Generate tasks for parameter recovery studies using multiprocessing.

    Parameters:
        n_subjects (int)         - Number of subjects
        n_tasks_per_subject (int)- Number of tasks per subject
        task_params (dict)       - Task generation parameters
        blocks (list)            - Optional block specifications for simulate_cpt
        n_processes (int)        - Number of parallel processes (default: cpu_count - 1)

    Returns:
        tasks (list) - List of task lists, tasks[i][j] is task j for subject i
    """
    # Build job list
    jobs = []
    for s in range(n_subjects):
        for t in range(n_tasks_per_subject):
            job_idx = len(jobs)
            seed = hash((s, t)) % (2**32)
            jobs.append((job_idx, s, t, task_params, blocks, seed))

    # Run jobs in parallel
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    n_jobs = len(jobs)
    print(f"Generating {n_jobs} tasks with {n_processes} processes...")

    report_interval = max(1, n_jobs // 20)
    results_list = [None] * n_jobs

    with mp.Pool(processes=n_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_task_job, jobs)):
            results_list[result['idx']] = result
            if (i + 1) % report_interval == 0 or i == n_jobs - 1:
                print(f"  Progress: {i + 1}/{n_jobs} tasks generated ({100*(i+1)/n_jobs:.0f}%)")

    # Reconstruct nested list structure
    tasks = [[] for _ in range(n_subjects)]
    for result in results_list:
        tasks[result['s']].append(result['task'])

    return tasks


# --- Parallel subject simulation ---

_sim_worker_config = None

def _init_sim_worker(param_ranges, fixed_params):
    """Initialize simulation worker process with shared configuration."""
    global _sim_worker_config
    _sim_worker_config = {
        'param_ranges': param_ranges,
        'fixed_params': fixed_params,
    }


def _worker_sim_job(args):
    """Worker function for parallel simulation."""
    job_idx, s, t, r, task_obs, task_noise_sd, task_hazard, true_param_vals, seed = args
    global _sim_worker_config

    np.random.seed(seed)

    param_ranges = _sim_worker_config['param_ranges']
    fixed_params = _sim_worker_config['fixed_params']

    # Assemble parameters for this subject
    subj_params = deepcopy(DEFAULT_PARAMS_SUBJ)
    for i, param_name in enumerate(param_ranges.keys()):
        subj_params[param_name] = true_param_vals[i]
    if fixed_params:
        subj_params.update(fixed_params)

    # Use task noise_sd and hazard unless we're fitting those
    if 'noise_sd' not in param_ranges:
        subj_params['noise_sd'] = task_noise_sd
    if 'hazard' not in param_ranges:
        subj_params['hazard'] = task_hazard

    # Run simulation
    subj = Subject()
    subj = simulate_subject(subj, task_obs, subj_params)

    return {'idx': job_idx, 's': s, 't': t, 'r': r, 'subj': subj}


def simulate_subjects_parallel(param_ranges, tasks, n_reps=1, fixed_params=None, n_processes=None):
    """
    Create subjects with known parameter values and simulate behavior using multiprocessing.

    Parameters:
        param_ranges (dict) - {param_name: (min, max)} for parameters to vary
        tasks (list)        - List of task lists, tasks[s][t] is task t for subject s
        n_reps (int)        - Number of simulation repetitions per subject-task
        fixed_params (dict) - Parameters held constant across subjects
        n_processes (int)   - Number of parallel processes (default: cpu_count - 1)

    Returns:
        subjects (list)     - Nested list subjects[s][t][r] of Subject instances
        true_params (dict)  - {param_name: array of true values per subject}
    """
    n_subjects = len(tasks)
    n_tasks = len(tasks[0]) if isinstance(tasks[0], list) else 1

    # Generate true parameter values for each subject
    true_params = {}
    for param_name, (lb, ub) in param_ranges.items():
        true_params[param_name] = np.random.uniform(lb, ub, n_subjects)

    # Build job list
    jobs = []
    for s in range(n_subjects):
        true_param_vals = [true_params[name][s] for name in param_ranges.keys()]

        for t in range(n_tasks):
            task = tasks[s][t] if isinstance(tasks[s], list) else tasks[s]

            for r in range(n_reps):
                job_idx = len(jobs)
                seed = hash((s, t, r)) % (2**32)
                jobs.append((job_idx, s, t, r, task.obs, task.noise_sd, task.hazard, true_param_vals, seed))

    # Run jobs in parallel
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    n_jobs = len(jobs)
    print(f"Simulating {n_jobs} subjects with {n_processes} processes...")

    report_interval = max(1, n_jobs // 20)
    results_list = [None] * n_jobs

    with mp.Pool(processes=n_processes, initializer=_init_sim_worker,
                 initargs=(param_ranges, fixed_params)) as pool:

        for i, result in enumerate(pool.imap_unordered(_worker_sim_job, jobs)):
            results_list[result['idx']] = result
            if (i + 1) % report_interval == 0 or i == n_jobs - 1:
                print(f"  Progress: {i + 1}/{n_jobs} simulations completed ({100*(i+1)/n_jobs:.0f}%)")

    # Reconstruct nested list structure
    subjects = [[[] for _ in range(n_tasks)] for _ in range(n_subjects)]
    for result in results_list:
        subjects[result['s']][result['t']].append(result['subj'])

    return subjects, true_params
