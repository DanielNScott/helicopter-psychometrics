"""
Maximum likelihood estimation for changepoint task subject parameters.

Provides functions for:
1. Simulating subjects with known parameters
2. Fitting parameters via MLE with multiprocessing
3. Analyzing recovery statistics (variance decomposition)
"""
from configs import *
from changepoint.subjects import Subject, simulate_subject, get_beliefs, DEFAULT_PARAMS_SUBJ
from changepoint.tasks import simulate_cpt
from changepoint.multiproc import generate_tasks_parallel, simulate_subjects_parallel
from analysis.analysis import fit_linear_models, build_design_matrix, get_model_terms, terms_to_params

import scipy.optimize as opt
from scipy.stats import norm
from copy import deepcopy
import multiprocessing as mp

import pickle
import time
import os

RESULTS_DIR = './data/'

# Parameter bounds for optimization (defaults populated from DEFAULT_PARAMS_SUBJ)
_BOUNDS_SPEC = {
    'beta_pe':         {'lb': 0.0,   'ub': 0.8},
    'beta_cpp':        {'lb': 0.0,   'ub': 1.0},
    'beta_ru':         {'lb': 0.0,   'ub': 1.0},
    'mix':             {'lb': 0.01,  'ub': 1.0},
    'hazard':          {'lb': 0.02,  'ub': 0.8},
    'noise_sd':        {'lb': 5.0,   'ub': 50.0},
    'loc':             {'lb': -40.0, 'ub': 40.0},
    'unc':             {'lb': 0.1,   'ub': 5.0},
    'drift':           {'lb': 0.0,   'ub': 10.0},
    'init_state_est':  {'lb': 0.0,   'ub': 300.0},
    'init_runlen_est': {'lb': 0.01,  'ub': 50.0},
    'noise_sd_update': {'lb': 0.0,   'ub': 10.0},
    'noise_sd_lr':     {'lb': 0.0,   'ub': 0.2},
    'init_relunc':     {'lb': 0.01,  'ub': 0.5},
    'ud':              {'lb': 0.1,   'ub': 10.0},
    'limit_updates':   {'lb': False, 'ub': True},
    'clip':            {'lb': False, 'ub': True},
}

# Build PARAM_BOUNDS with defaults from DEFAULT_PARAMS_SUBJ
PARAM_BOUNDS = {
    name: {'default': DEFAULT_PARAMS_SUBJ[name], **bounds}
    for name, bounds in _BOUNDS_SPEC.items()
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
    n_sims = n_subjects * n_tasks * n_reps
    report_interval = max(1, n_sims // 20)
    sim_count = 0

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

                sim_count += 1
                if sim_count % report_interval == 0 or sim_count == n_sims:
                    print(f"  Progress: {sim_count}/{n_sims} simulations completed ({100*sim_count/n_sims:.0f}%)")

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


def fit_single_ols(subj, task, opt_param_names, fixed_params=None, model='model-pe-cpp-ru'):
    """
    Fit beta parameters for a single subject-task combination using OLS.

    Beliefs (cpp, ru) are re-inferred using default parameters, matching
    how real subject data would be processed.

    Parameters:
        subj (Subject)        - Subject with responses
        task (ChangepointTask)- Task data (for noise_sd, hazard, new_blk)
        opt_param_names (list)- Parameter names to extract (beta_ prefixed)
        fixed_params (dict)   - Fixed parameter values (unused, for interface compatibility)
        model (str)           - Model name matching fit_linear_models

    Returns:
        estimates (array) - Fitted parameter values in order of opt_param_names
        sse (float)       - Sum of squared errors
    """
    import statsmodels.api as sm

    # Re-infer beliefs using default parameters (as we would for real data)
    noise_sd = task.noise_sd if not np.isscalar(task.noise_sd) else np.full(len(task.obs), task.noise_sd)
    subj.beliefs = get_beliefs(subj.responses, task.obs, task.new_blk, task.hazard, noise_sd)

    # Build design matrix
    pe  = subj.responses.pe
    up  = subj.responses.update
    cpp = subj.beliefs.cpp
    ru  = subj.beliefs.relunc
    design, short_terms = build_design_matrix(pe, cpp, ru, model)

    # Map short term names to beta-prefixed parameter names
    # short_terms includes intercept 'c'; strip it for terms_to_params, then prepend
    term_names = ['c'] + terms_to_params(short_terms[1:])

    # Validate requested params exist in this model
    valid_params = set(term_names)
    invalid = set(opt_param_names) - valid_params
    if invalid:
        raise ValueError(f"Model {model} has params {valid_params}. Invalid: {invalid}")

    # Fit model
    results = sm.OLS(up, design).fit()

    # Extract requested parameters in order
    estimates = []
    for name in opt_param_names:
        idx = term_names.index(name)
        estimates.append(results.params[idx])

    sse = results.ssr
    return np.array(estimates), sse


def fit_single(subj, task, opt_param_names, fixed_params=None, seed=None,
               fit_method='mle', model='model-pe-cpp-ru'):
    """
    Fit parameters for a single subject-task combination.

    Parameters:
        subj (Subject)        - Subject with responses
        task (ChangepointTask)- Task data
        opt_param_names (list)- Names of parameters to fit
        fixed_params (dict)   - Fixed parameter values
        seed (int)            - Random seed for MLE initialization
        fit_method (str)      - 'mle' for maximum likelihood, 'ols' for ordinary least squares
        model (str)           - Model name for OLS fitting

    Returns:
        estimates (array) - Fitted parameter values
        score (float)     - Negative log-likelihood (MLE) or sum of squared errors (OLS)
    """
    if fit_method == 'ols':
        return fit_single_ols(subj, task, opt_param_names, fixed_params, model=model)

    # MLE fitting
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

def _init_worker(opt_param_names, fixed_params, fit_method, model='model-pe-cpp-ru'):
    """Initialize worker process with shared configuration."""
    global _worker_config
    _worker_config = {
        'opt_param_names': opt_param_names,
        'fixed_params': fixed_params,
        'fit_method': fit_method,
        'model': model,
    }


def _worker_fit_job(args):
    """Worker function for parallel fitting."""
    job_idx, subj, task, seed = args
    global _worker_config

    try:
        estimates, score = fit_single(
            subj, task,
            _worker_config['opt_param_names'],
            _worker_config['fixed_params'],
            seed,
            _worker_config['fit_method'],
            _worker_config['model'],
        )
        return {'success': True, 'estimates': estimates, 'score': score, 'idx': job_idx}
    except Exception as e:
        return {'success': False, 'error': str(e), 'idx': job_idx}


def fit_parameters(subjects, tasks, opt_param_names, n_refits=5, n_processes=None,
                   fixed_params=None, fit_method='mle', model='model-pe-cpp-ru'):
    """
    Fit parameters for multiple subjects with multiprocessing.

    Parameters:
        subjects (list)       - Nested list subjects[s][t][r] of Subject instances
        tasks (list)          - List of task lists, tasks[s][t] is task t for subject s
        opt_param_names (list)- Parameter names to optimize
        n_refits (int)        - Number of fitting repetitions per subject-task-rep
        n_processes (int)     - Number of parallel processes
        fixed_params (dict)   - Parameters held fixed during optimization
        fit_method (str)      - 'mle' for maximum likelihood, 'ols' for ordinary least squares

    Returns:
        results (dict) with:
            'ests' (ndarray)    - Shape [n_subj, n_tasks, n_reps, n_refits, n_params]
            'scores' (ndarray)  - Shape [n_subj, n_tasks, n_reps, n_refits] (NLL or SSE)
            'param_names' (list)
            'fit_method' (str)
    """
    # For OLS, n_refits is forced to 1 since the solution is deterministic
    if fit_method == 'ols':
        if n_refits != 1:
            print(f"OLS fitting is deterministic; setting n_refits=1 (was {n_refits})")
            n_refits = 1

    n_subj = len(subjects)
    n_tasks = len(subjects[0])
    n_reps = len(subjects[0][0])
    n_params = len(opt_param_names)

    # Pre-allocate results arrays
    ests = np.full((n_subj, n_tasks, n_reps, n_refits, n_params), np.nan)
    scores = np.full((n_subj, n_tasks, n_reps, n_refits), np.nan)

    # Build job list with index for tracking
    jobs = []
    job_map = []
    for s in range(n_subj):
        for t in range(n_tasks):
            for r in range(n_reps):
                for f in range(n_refits):
                    job_idx = len(jobs)
                    seed = hash((s, t, r, f)) % (2**32)
                    jobs.append((job_idx, deepcopy(subjects[s][t][r]), deepcopy(tasks[s][t]), seed))
                    job_map.append((s, t, r, f))

    # Run jobs in parallel
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    n_jobs = len(jobs)
    print(f"Fitting {n_jobs} jobs with {n_processes} processes...")

    # Use imap_unordered for progress reporting
    results_list = [None] * n_jobs
    report_interval = max(1, n_jobs // 20)  # Report ~20 times

    with mp.Pool(processes=n_processes, initializer=_init_worker,
                 initargs=(opt_param_names, fixed_params, fit_method, model)) as pool:

        for i, result in enumerate(pool.imap_unordered(_worker_fit_job, jobs)):
            results_list[result['idx']] = result
            if (i + 1) % report_interval == 0 or i == n_jobs - 1:
                print(f"  Progress: {i + 1}/{n_jobs} fits completed ({100*(i+1)/n_jobs:.0f}%)")

    # Collect results
    n_success = 0
    for result, (s, t, r, f) in zip(results_list, job_map):
        if result['success']:
            ests[s, t, r, f, :] = result['estimates']
            scores[s, t, r, f] = result['score']
            n_success += 1
        else:
            print(f"Fit failed for subject {s}, task {t}, rep {r}, refit {f}: {result['error']}")

    print(f"Completed {n_success}/{len(jobs)} fits successfully")

    return {'ests': ests, 'scores': scores, 'param_names': opt_param_names, 'fit_method': fit_method}


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

        # Skip params that don't have a generative truth (e.g. beta_prod)
        if param_name not in true_params:
            continue

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
                       param_ranges=None, task_params=None, blocks=None, fixed_params=None,
                       n_processes=None, fit_method='mle', model='model-pe-cpp-ru'):
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
        fit_method (str)           - 'mle' for maximum likelihood, 'ols' for ordinary least squares

    Returns:
        dict with 'true_params', 'results', 'analysis', 'config', 'subjects', 'tasks'
    """
    # Default parameter ranges from PARAM_BOUNDS
    if param_ranges is None:
        param_ranges = {name: (PARAM_BOUNDS[name]['lb'], PARAM_BOUNDS[name]['ub'])
                        for name in param_names}

    # Generate tasks (parallel)
    print(f"Setting up {n_subjects} x {n_tasks_per_subject} tasks...")
    tasks = generate_tasks_parallel(n_subjects, n_tasks_per_subject, task_params, blocks, n_processes)

    # Simulate subjects with known parameters (parallel)
    print(f"Setting up {n_subjects} x {n_tasks_per_subject} x {n_reps} subject behaviors...")
    subjects, true_params = simulate_subjects_parallel(param_ranges, tasks, n_reps, fixed_params, n_processes)

    # Fit parameters
    n_fits = n_subjects * n_tasks_per_subject * n_reps * n_refits
    print(f"Fitting parameters {param_names} ({n_fits} total fits, method={fit_method})...")
    results = fit_parameters(subjects, tasks, param_names, n_refits, n_processes, fixed_params,
                             fit_method, model)

    # Analyze recovery
    analysis = analyze_recovery(true_params, results)

    # Print summary
    print("\nRecovery Summary:")
    for param_name in param_names:
        if param_name not in analysis:
            continue
        a = analysis[param_name]
        print(f"  {param_name}: r={a['correlation']:.3f}, bias={a['mean_bias']:.3f}, "
              f"std_fit={np.sqrt(a['var_fit']):.3f}, std_rep={np.sqrt(a['var_rep']):.3f}, "
              f"std_task={np.sqrt(a['var_task']):.3f}")

    return {
        'true_params': true_params,
        'results': results,
        'analysis': analysis,
        'subjects': subjects,
        'tasks': tasks,
        'config': {
            'param_names': param_names,
            'n_subjects': n_subjects,
            'n_tasks_per_subject': n_tasks_per_subject,
            'n_reps': n_reps,
            'n_refits': n_refits,
            'param_ranges': param_ranges,
            'fit_method': fit_method,
            'model': model,
        }
    }

def save_recovery(result, filename='recovery'):
    """Save parameter recovery result to timestamped pickle file.

    Parameters:
        result (dict) - Output from parameter_recovery
        filename (str) - Base filename (without extension)

    Returns:
        filepath (str) - Full path to saved file
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    datestring = time.strftime("%Y-%m-%d-%H-%M-%S")
    filepath = f"{RESULTS_DIR}{filename}_{datestring}.pkl"

    with open(filepath, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved recovery results to {filepath}")
    return filepath


def load_recovery(filename='recovery'):
    """Load most recent parameter recovery result from pickle file.

    Parameters:
        filename (str) - Base filename (without extension)

    Returns:
        result (dict) - Loaded parameter recovery result
    """
    file_list = [f for f in os.listdir(RESULTS_DIR) if f.startswith(f"{filename}_") and f.endswith('.pkl')]

    if not file_list:
        raise FileNotFoundError(f"No recovery files found matching '{filename}_*.pkl' in {RESULTS_DIR}")

    latest_file = max(file_list, key=lambda x: os.path.getctime(os.path.join(RESULTS_DIR, x)))
    filepath = os.path.join(RESULTS_DIR, latest_file)

    with open(filepath, 'rb') as f:
        result = pickle.load(f)

    print(f"Loaded recovery results from {filepath}")
    return result
