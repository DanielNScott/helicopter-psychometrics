"""
Maximum likelihood estimation for changepoint task subject parameters.

Provides functions for:
1. Simulating subjects with known parameters
2. Fitting parameters via MLE with multiprocessing
3. Analyzing recovery statistics (variance decomposition)
"""
from configs import *
from subjects import Subject, simulate_subject, get_beliefs, DEFAULT_PARAMS_SUBJ
from tasks import simulate_cpt

import scipy.optimize as opt
from scipy.stats import norm
from copy import deepcopy
import multiprocessing as mp

# Parameter bounds for optimization
PARAM_BOUNDS = {
    'beta_pe':         {'default': 0.0,   'lb': 0.0,   'ub': 0.8},
    'beta_cpp':        {'default': 1.0,   'lb': 0.0,   'ub': 1.0},
    'beta_ru':         {'default': 1.0,   'lb': 0.0,   'ub': 1.0},
    'mix':             {'default': 1.0,   'lb': 0.01,  'ub': 1.0},
    'hazard':          {'default': 0.1,   'lb': 0.02,  'ub': 0.8},
    'noise_sd':        {'default': 10.0,  'lb': 5.0,   'ub': 50.0},
    'loc':             {'default': 0.0,   'lb': -40.0, 'ub': 40.0},
    'unc':             {'default': 1.0,   'lb': 0.1,   'ub': 5.0},
    'drift':           {'default': 0.0,   'lb': 0.0,   'ub': 10.0},
    'init_state_est':  {'default': 150.0, 'lb': 0.0,   'ub': 300.0},
    'init_runlen_est': {'default': 0.1,   'lb': 0.1,   'ub': 50.0},
    'noise_sd_update': {'default': 2.0,   'lb': 0.0,   'ub': 10.0},
    'limit_updates':   {'default': True,  'lb': False, 'ub': True},
    'clip':            {'default': True,  'lb': False, 'ub': True},
}

# Validate that PARAM_BOUNDS covers all subject parameters
_missing = set(DEFAULT_PARAMS_SUBJ.keys()) - set(PARAM_BOUNDS.keys())
if _missing:
    raise ValueError(f"PARAM_BOUNDS missing fields from DEFAULT_PARAMS_SUBJ: {_missing}")


# --- Simulation functions ---

def simulate_subjects(param_ranges, tasks, n_reps=1, fixed_params=None):
    """
    Create subjects with known parameter values and simulate behavior.

    Parameters:
        param_ranges (dict) - {param_name: (min, max)} for parameters to vary
        tasks (list)        - List of task lists, tasks[s][t] is task t for subject s
        n_reps (int)        - Number of simulation repetitions per subject-task
        fixed_params (dict) - Parameters held constant across subjects

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

    # Simulate each subject on each task, multiple times
    subjects = []
    for s in range(n_subjects):

        # Assemble parameters for this subject, starting from subject defaults
        subj_params = deepcopy(DEFAULT_PARAMS_SUBJ)
        for param_name in param_ranges:
            subj_params[param_name] = true_params[param_name][s]
        if fixed_params:
            subj_params.update(fixed_params)

        subj_tasks = []
        for t in range(n_tasks):
            task = tasks[s][t] if isinstance(tasks[s], list) else tasks[s]

            # Use task noise_sd and hazard unless we're fitting those
            task_params = deepcopy(subj_params)
            if 'noise_sd' not in param_ranges:
                task_params['noise_sd'] = task.noise_sd
            if 'hazard' not in param_ranges:
                task_params['hazard'] = task.hazard

            subj_reps = []
            for r in range(n_reps):
                subj = Subject()
                subj = simulate_subject(subj, task.obs, task_params)
                subj_reps.append(subj)

            subj_tasks.append(subj_reps)
        subjects.append(subj_tasks)

    return subjects, true_params


def generate_tasks(n_subjects, n_tasks_per_subject, task_params=None, blocks=None):
    """
    Generate tasks for parameter recovery studies.

    Parameters:
        n_subjects (int)         - Number of subjects
        n_tasks_per_subject (int)- Number of tasks per subject
        task_params (dict)       - Task generation parameters
        blocks (list)            - Optional block specifications for simulate_cpt

    Returns:
        tasks (list) - List of task lists, tasks[i][j] is task j for subject i
    """
    # Default task parameters
    if task_params is None:
        task_params = {
            'ntrials':  200,
            'noise_sd': 25,
            'hazard':   0.1,
            'bnds_ls':  [0, 300],
            'bnds_obs': [0, 300],
        }

    # Generate tasks
    tasks = []
    for s in range(n_subjects):
        subj_tasks = [simulate_cpt(task_params, blocks=blocks) for _ in range(n_tasks_per_subject)]
        tasks.append(subj_tasks)

    return tasks


# --- Fitting functions ---

def get_negloglike(subj, task, opt_param_names, opt_param_vals, fixed_params=None):
    """
    Compute negative log-likelihood for a subject's behavior given parameters.

    Parameters:
        subj (Subject)        - Subject with responses
        task (ChangepointTask)- Task data
        opt_param_names (list)- Names of parameters being optimized
        opt_param_vals (array)- Current values for optimized parameters
        fixed_params (dict)   - Fixed parameter values

    Returns:
        nll (float) - Negative log-likelihood
    """
    # Assemble full parameter set
    params = deepcopy(DEFAULT_PARAMS_SUBJ)
    for name, val in zip(opt_param_names, opt_param_vals):
        params[name] = val
    if fixed_params:
        params.update(fixed_params)

    # Use task hazard unless optimizing it
    if 'hazard' not in opt_param_names:
        params['hazard'] = task.hazard

    # Build noise_sd vector for get_beliefs
    # If optimizing noise_sd, use the optimized scalar uniformly
    # Otherwise, use the task's noise_sd vector (which may vary by trial/block)
    if 'noise_sd' in opt_param_names:
        noise_sd_vec = np.full(len(task.obs), params['noise_sd'])
    elif np.isscalar(task.noise_sd):
        noise_sd_vec = np.full(len(task.obs), task.noise_sd)
    else:
        noise_sd_vec = task.noise_sd

    # Infer beliefs under these parameters
    beliefs = get_beliefs(subj.responses, task.obs, task.new_blk, params['hazard'], noise_sd_vec, params=params)

    # Residuals: actual update minus model-predicted update
    residuals = subj.responses.update - beliefs.model_up

    # Heteroscedastic noise model: sigma scales with |PE|
    update_sd_int, update_sd_slope = params['noise_sd_update'], 0.1
    sigma = np.maximum(update_sd_int + update_sd_slope * np.abs(subj.responses.pe), 1e-4)

    # Gaussian log-likelihood
    log_probs = -0.5 * (residuals / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    nll = -np.sum(log_probs)

    # Handle numerical issues
    if not np.isfinite(nll):
        return 1e10

    return nll


def get_param_guess(opt_param_names):
    """Generate random initial parameter values from truncated normal."""
    guesses = []
    for name in opt_param_names:
        bounds = PARAM_BOUNDS[name]
        mu, lb, ub = bounds['default'], bounds['lb'], bounds['ub']
        sigma = (ub - lb) / 4

        # Sample from truncated normal
        tmax = norm.cdf(ub, mu, sigma)
        tmin = norm.cdf(lb, mu, sigma)
        sample = norm.ppf((tmax - tmin) * np.random.rand() + tmin, mu, sigma)
        guesses.append(np.clip(sample, lb, ub))

    return np.array(guesses)


def fit_single(subj, task, opt_param_names, fixed_params=None, seed=None):
    """
    Fit parameters for a single subject-task combination.

    Returns:
        estimates (array) - Fitted parameter values
        nll (float)       - Final negative log-likelihood
    """
    if seed is not None:
        np.random.seed(seed)

    # Initial guess and bounds
    guess = get_param_guess(opt_param_names)
    bounds = [(PARAM_BOUNDS[name]['lb'], PARAM_BOUNDS[name]['ub']) for name in opt_param_names]

    # Objective function
    def objective(params):
        return get_negloglike(subj, task, opt_param_names, params, fixed_params)

    # Run optimizer
    result = opt.minimize(objective, guess, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 500, 'disp': False})

    return result.x, result.fun


# Global for worker initialization
_worker_config = None

def _init_worker(opt_param_names, fixed_params):
    """Initialize worker process with shared configuration."""
    global _worker_config
    _worker_config = {'opt_param_names': opt_param_names, 'fixed_params': fixed_params}


def _worker_fit_job(args):
    """Worker function for parallel fitting."""
    subj, task, seed = args
    global _worker_config

    try:
        estimates, nll = fit_single(subj, task, _worker_config['opt_param_names'],
                                    _worker_config['fixed_params'], seed)
        return {'success': True, 'estimates': estimates, 'nll': nll}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def fit_parameters(subjects, tasks, opt_param_names, n_refits=5, n_processes=None, fixed_params=None):
    """
    Fit parameters for multiple subjects with multiprocessing.

    Parameters:
        subjects (list)       - Nested list subjects[s][t][r] of Subject instances
        tasks (list)          - List of task lists, tasks[s][t] is task t for subject s
        opt_param_names (list)- Parameter names to optimize
        n_refits (int)        - Number of fitting repetitions per subject-task-rep
        n_processes (int)     - Number of parallel processes
        fixed_params (dict)   - Parameters held fixed during optimization

    Returns:
        results (dict) with:
            'ests' (ndarray) - Shape [n_subj, n_tasks, n_reps, n_refits, n_params]
            'nlls' (ndarray) - Shape [n_subj, n_tasks, n_reps, n_refits]
            'param_names' (list)
    """
    n_subj = len(subjects)
    n_tasks = len(subjects[0])
    n_reps = len(subjects[0][0])
    n_params = len(opt_param_names)

    # Pre-allocate results arrays
    ests = np.full((n_subj, n_tasks, n_reps, n_refits, n_params), np.nan)
    nlls = np.full((n_subj, n_tasks, n_reps, n_refits), np.nan)

    # Build job list
    jobs = []
    job_map = []
    for s in range(n_subj):
        for t in range(n_tasks):
            for r in range(n_reps):
                for f in range(n_refits):
                    seed = hash((s, t, r, f)) % (2**32)
                    jobs.append((deepcopy(subjects[s][t][r]), deepcopy(tasks[s][t]), seed))
                    job_map.append((s, t, r, f))

    # Run jobs in parallel
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    print(f"Fitting {len(jobs)} jobs with {n_processes} processes...")
    with mp.Pool(processes=n_processes, initializer=_init_worker,
                 initargs=(opt_param_names, fixed_params)) as pool:
        results_list = pool.map(_worker_fit_job, jobs)

    # Collect results
    n_success = 0
    for result, (s, t, r, f) in zip(results_list, job_map):
        if result['success']:
            ests[s, t, r, f, :] = result['estimates']
            nlls[s, t, r, f] = result['nll']
            n_success += 1
        else:
            print(f"Fit failed for subject {s}, task {t}, rep {r}, refit {f}: {result['error']}")

    print(f"Completed {n_success}/{len(jobs)} fits successfully")

    return {'ests': ests, 'nlls': nlls, 'param_names': opt_param_names}


# --- Analysis functions ---

def recovery_stats(results):
    """
    Compute variance decomposition for parameter recovery.

    Parameters:
        results (dict) - Output from fit_parameters with 'ests' array

    Returns:
        mean (ndarray)     - Mean estimates, shape [n_subj, n_params]
        var_fit (ndarray)  - Variance over refits (optimizer noise), shape [n_subj, n_params]
        var_rep (ndarray)  - Variance over reps (behavioral stochasticity), shape [n_subj, n_params]
        var_task (ndarray) - Variance over tasks (task realization effects), shape [n_subj, n_params]
        std_tot (ndarray)  - Total standard deviation, shape [n_subj, n_params]
    """
    ests = results['ests']  # Shape: [n_subj, n_tasks, n_reps, n_refits, n_params]

    # Mean parameter estimates (over tasks, reps, and refits)
    mean = np.nanmean(ests, axis=(1, 2, 3))

    # var_fit: variance over refits, averaged over reps and tasks
    var_fit = np.nanmean(np.nanvar(ests, axis=3), axis=(1, 2))

    # Mean over refits first
    mean_over_fits = np.nanmean(ests, axis=3)  # [n_subj, n_tasks, n_reps, n_params]

    # var_rep: variance over reps (of mean over fits), averaged over tasks
    var_rep = np.nanmean(np.nanvar(mean_over_fits, axis=2), axis=1)

    # Mean over reps
    mean_over_reps = np.nanmean(mean_over_fits, axis=2)  # [n_subj, n_tasks, n_params]

    # var_task: variance over tasks (of mean over reps and fits)
    var_task = np.nanvar(mean_over_reps, axis=1)

    # Total standard deviation
    std_tot = np.sqrt(var_fit + var_rep + var_task)

    return mean, var_fit, var_rep, var_task, std_tot


def analyze_recovery(true_params, results):
    """
    Analyze parameter recovery performance.

    Parameters:
        true_params (dict) - {param_name: array of true values}
        results (dict)     - Output from fit_parameters

    Returns:
        analysis (dict) - Recovery statistics per parameter
    """
    param_names = results['param_names']
    mean_ests, var_fit, var_rep, var_task, std_tot = recovery_stats(results)

    analysis = {}
    for p, param_name in enumerate(param_names):
        true_vals = true_params[param_name]
        recovered_vals = mean_ests[:, p]

        # Filter out any NaN values
        valid = ~np.isnan(recovered_vals)
        true_valid = true_vals[valid]
        rec_valid = recovered_vals[valid]

        if len(rec_valid) > 1:
            correlation = np.corrcoef(true_valid, rec_valid)[0, 1]
            bias = np.mean(rec_valid - true_valid)
            rmse = np.sqrt(np.mean((rec_valid - true_valid)**2))
        else:
            correlation, bias, rmse = np.nan, np.nan, np.nan

        analysis[param_name] = {
            'correlation': correlation,
            'mean_bias': bias,
            'rmse': rmse,
            'var_fit': np.mean(var_fit[:, p]),
            'var_rep': np.mean(var_rep[:, p]),
            'var_task': np.mean(var_task[:, p]),
            'std_tot': np.mean(std_tot[:, p]),
            'true_values': true_vals,
            'recovered_values': recovered_vals,
        }

    return analysis


# --- High-level convenience function ---

def parameter_recovery(param_names, n_subjects=10, n_tasks_per_subject=5, n_reps=3, n_refits=3,
                       param_ranges=None, task_params=None, blocks=None, fixed_params=None, n_processes=None):
    """
    Run a complete parameter recovery study.

    Parameters:
        param_names (list)         - Parameters to vary and recover
        n_subjects (int)           - Number of simulated subjects
        n_tasks_per_subject (int)  - Tasks per subject
        n_reps (int)               - Simulation repetitions per subject-task
        n_refits (int)             - Fitting repetitions per subject-task-rep
        param_ranges (dict)        - {param_name: (min, max)}, uses PARAM_BOUNDS defaults if None
        task_params (dict)         - Task generation parameters
        blocks (list)              - Optional block specifications for tasks
        fixed_params (dict)        - Fixed parameter values
        n_processes (int)          - Number of parallel processes

    Returns:
        dict with 'true_params', 'results', 'analysis', 'config'
    """
    # Default parameter ranges from PARAM_BOUNDS
    if param_ranges is None:
        param_ranges = {name: (PARAM_BOUNDS[name]['lb'], PARAM_BOUNDS[name]['ub'])
                        for name in param_names}

    # Generate tasks
    print(f"Generating {n_subjects} x {n_tasks_per_subject} tasks...")
    tasks = generate_tasks(n_subjects, n_tasks_per_subject, task_params, blocks)

    # Simulate subjects with known parameters
    print(f"Simulating {n_subjects} x {n_tasks_per_subject} x {n_reps} subject behaviors...")
    subjects, true_params = simulate_subjects(param_ranges, tasks, n_reps, fixed_params)

    # Fit parameters
    n_fits = n_subjects * n_tasks_per_subject * n_reps * n_refits
    print(f"Fitting parameters {param_names} ({n_fits} total fits)...")
    results = fit_parameters(subjects, tasks, param_names, n_refits, n_processes, fixed_params)

    # Analyze recovery
    analysis = analyze_recovery(true_params, results)

    # Print summary
    print("\nRecovery Summary:")
    for param_name in param_names:
        a = analysis[param_name]
        print(f"  {param_name}: r={a['correlation']:.3f}, bias={a['mean_bias']:.3f}, "
              f"std_fit={np.sqrt(a['var_fit']):.3f}, std_rep={np.sqrt(a['var_rep']):.3f}, "
              f"std_task={np.sqrt(a['var_task']):.3f}")

    return {
        'true_params': true_params,
        'results': results,
        'analysis': analysis,
        'config': {
            'param_names': param_names,
            'n_subjects': n_subjects,
            'n_tasks_per_subject': n_tasks_per_subject,
            'n_reps': n_reps,
            'n_refits': n_refits,
            'param_ranges': param_ranges,
        }
    }


def verify_block_recovery():
    """
    Verify parameter recovery with block-structured tasks.

    Uses 4 blocks of 120 trials each with noise_sd alternating between 10 and 25.
    Fits beta_cpp and beta_ru for 10 subjects.
    """
    blocks = [
        {'ntrials': 120, 'noise_sd': 10},
        {'ntrials': 120, 'noise_sd': 25},
        {'ntrials': 120, 'noise_sd': 10},
        {'ntrials': 120, 'noise_sd': 25},
    ]

    result = parameter_recovery(
        param_names=['mix'],
        n_subjects=10,
        n_tasks_per_subject=5,
        n_reps=5,
        n_refits=1,
        blocks=blocks,
    )

    return result
