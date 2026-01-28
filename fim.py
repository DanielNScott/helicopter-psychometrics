"""Fisher information analysis for parameter estimation.

Provides functions to compute and analyze Fisher information matrices
for the n0 regression model, at both aggregate and per-task levels.
"""
from configs import *
from recovery import recovery_stats

def compute_regressor_fim(subjects, tasks, noise_var=None):
    """
    Compute Fisher information approximation from regressor covariance.

    For OLS with model: update = c + beta_cpp*X1 + beta_ru*X2
    where X1 = pe*cpp, X2 = pe*ru*(1-cpp), the Fisher information matrix
    is proportional to X'X / noise_var.

    Parameters:
        subjects (list) - Nested list subjects[s][t][r] or flat list of Subject instances
        tasks (list)    - Corresponding task list (same structure as subjects)
        noise_var (float) - Noise variance for scaling. If None, uses residual variance.

    Returns:
        dict with:
            'fim' (ndarray)          - Per-trial Fisher information matrix [2, 2]
            'fim_inv' (ndarray)      - Inverse per-trial FIM [2, 2]
            'eigenvalues' (ndarray)  - FIM eigenvalues (largest first)
            'eigenvectors' (ndarray) - FIM eigenvectors as columns
            'condition_number' (float) - Ratio of largest to smallest eigenvalue
            'regressor_corr' (float) - Correlation between regressors
            'n_trials' (int)         - Total trials used
            'param_names' (list)     - ['beta_cpp', 'beta_ru']
    """
    # Flatten nested subject/task lists if needed
    subj_list = []
    task_list = []

    if isinstance(subjects[0], list):
        for s in range(len(subjects)):
            for t in range(len(subjects[s])):
                for r in range(len(subjects[s][t])):
                    subj_list.append(subjects[s][t][r])
                    task_list.append(tasks[s][t])
    else:
        subj_list = subjects
        task_list = tasks

    # Collect regressors from all subjects/tasks
    X1_all, X2_all, resid_all = [], [], []

    for subj, task in zip(subj_list, task_list):
        pe = subj.responses.pe
        cpp = subj.beliefs.cpp
        ru = subj.beliefs.relunc
        up = subj.responses.update

        X1 = pe * cpp
        X2 = pe * ru * (1 - cpp)

        X1_all.append(X1)
        X2_all.append(X2)

        # Compute residuals for noise variance estimate
        X = np.column_stack([np.ones(len(pe)), X1, X2])
        beta_hat = np.linalg.lstsq(X, up, rcond=None)[0]
        resid = up - X @ beta_hat
        resid_all.append(resid)

    X1_all = np.concatenate(X1_all)
    X2_all = np.concatenate(X2_all)
    resid_all = np.concatenate(resid_all)
    n_trials = len(X1_all)

    # Estimate noise variance from residuals if not provided
    if noise_var is None:
        noise_var = np.var(resid_all)

    # Build design matrix (excluding intercept for FIM of beta parameters)
    X = np.column_stack([X1_all, X2_all])

    # Fisher information per trial: X'X / (n_trials * noise_var)
    fim = (X.T @ X) / (n_trials * noise_var)

    # Inverse FIM (asymptotic covariance scaled by n_trials)
    fim_inv = np.linalg.inv(fim)

    # Eigendecomposition of FIM (sorted descending)
    eigenvalues, eigenvectors = np.linalg.eigh(fim)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Condition number (ratio of largest to smallest eigenvalue)
    condition_number = eigenvalues[0] / eigenvalues[-1]

    # Regressor correlation
    regressor_corr = np.corrcoef(X1_all, X2_all)[0, 1]

    return {
        'fim': fim,
        'fim_inv': fim_inv,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'condition_number': condition_number,
        'regressor_corr': regressor_corr,
        'n_trials': n_trials,
        'noise_var': noise_var,
        'param_names': ['beta_cpp', 'beta_ru'],
    }


def analyze_error_covariance(true_params, results):
    """
    Analyze estimation error covariance structure.

    Computes the covariance of estimation errors across subjects, which
    approximates the inverse Fisher information matrix. Eigendecomposition
    reveals constrained vs unconstrained parameter directions.

    Parameters:
        true_params (dict) - {param_name: array of true values per subject}
        results (dict)     - Output from fit_parameters

    Returns:
        dict with:
            'errors' (ndarray)      - Estimation errors [n_subj, n_params]
            'cov' (ndarray)         - Error covariance matrix [n_params, n_params]
            'corr' (ndarray)        - Error correlation matrix [n_params, n_params]
            'eigenvalues' (ndarray) - Eigenvalues (largest first)
            'eigenvectors' (ndarray)- Eigenvectors as columns [n_params, n_params]
            'param_names' (list)    - Parameter names in order
    """
    param_names = results['param_names']
    n_params = len(param_names)

    # Get mean estimates per subject (averaged over tasks, reps, refits)
    mean_ests, _, _, _, _ = recovery_stats(results)

    # Build error matrix [n_subj, n_params]
    errors = np.column_stack([
        mean_ests[:, p] - true_params[name]
        for p, name in enumerate(param_names)
    ])

    # Remove subjects with NaN estimates
    valid = ~np.any(np.isnan(errors), axis=1)
    errors_valid = errors[valid]

    # Compute covariance and correlation
    cov = np.cov(errors_valid, rowvar=False)
    stds = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stds, stds)

    # Eigendecomposition (sorted by descending eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return {
        'errors': errors,
        'cov': cov,
        'corr': corr,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'param_names': param_names,
    }


def compute_task_fim(subj, task, noise_var=None):
    """
    Compute Fisher information matrix for a single subject-task.

    For OLS with model: update = c + beta_cpp*X1 + beta_ru*X2
    where X1 = pe*cpp, X2 = pe*ru*(1-cpp).

    Parameters:
        subj (Subject) - Subject with responses and beliefs
        task (ChangepointTask) - Task data
        noise_var (float) - Noise variance. If None, estimated from residuals.

    Returns:
        dict with:
            'fim' (ndarray) - Per-trial FIM [2, 2]
            'condition_number' (float)
            'regressor_corr' (float)
            'regressor_var' (ndarray) - Variance of each regressor [2]
            'n_trials' (int)
    """
    pe = subj.responses.pe
    cpp = subj.beliefs.cpp
    ru = subj.beliefs.relunc
    up = subj.responses.update

    X1 = pe * cpp
    X2 = pe * ru * (1 - cpp)
    n_trials = len(pe)

    # Estimate noise variance from residuals if not provided
    if noise_var is None:
        X = np.column_stack([np.ones(n_trials), X1, X2])
        beta_hat = np.linalg.lstsq(X, up, rcond=None)[0]
        resid = up - X @ beta_hat
        noise_var = np.var(resid)

    # Build design matrix for beta parameters only
    X = np.column_stack([X1, X2])

    # Per-trial FIM
    fim = (X.T @ X) / (n_trials * noise_var)

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(fim)
    eigenvalues = np.sort(eigenvalues)[::-1]
    condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf

    # Regressor statistics
    regressor_var = np.array([np.var(X1), np.var(X2)])
    regressor_corr = np.corrcoef(X1, X2)[0, 1]

    return {
        'fim': fim,
        'eigenvalues': eigenvalues,
        'condition_number': condition_number,
        'regressor_corr': regressor_corr,
        'regressor_var': regressor_var,
        'n_trials': n_trials,
        'noise_var': noise_var,
    }


def compute_task_features(task):
    """
    Extract task features relevant to Fisher information.

    Parameters:
        task (ChangepointTask) - Task data

    Returns:
        dict with task summary statistics
    """
    n_trials = len(task.obs)
    n_changepoints = np.sum(task.cp)

    # Run lengths (trials between changepoints)
    cp_indices = np.where(task.cp)[0]
    if len(cp_indices) > 0:
        run_lengths = np.diff(np.concatenate([[0], cp_indices, [n_trials]]))
        mean_run_length = np.mean(run_lengths)
        std_run_length = np.std(run_lengths)
    else:
        mean_run_length = n_trials
        std_run_length = 0.0

    # Noise characteristics
    noise_sd = task.noise_sd
    if np.isscalar(noise_sd):
        mean_noise_sd = noise_sd
        std_noise_sd = 0.0
    else:
        mean_noise_sd = np.mean(noise_sd)
        std_noise_sd = np.std(noise_sd)

    return {
        'n_trials': n_trials,
        'n_changepoints': n_changepoints,
        'cp_rate': n_changepoints / n_trials,
        'mean_run_length': mean_run_length,
        'std_run_length': std_run_length,
        'hazard': task.hazard,
        'mean_noise_sd': mean_noise_sd,
        'std_noise_sd': std_noise_sd,
    }


def analyze_task_information(subjects, tasks, noise_var=None):
    """
    Compute per-task FIM statistics and task features.

    Parameters:
        subjects (list) - Nested list subjects[s][t][r] of Subject instances
        tasks (list) - List of task lists, tasks[s][t]
        noise_var (float) - Shared noise variance. If None, estimated per-task.

    Returns:
        pd.DataFrame with one row per task, columns for FIM metrics and task features
    """
    rows = []

    n_subj = len(subjects)
    n_tasks = len(subjects[0])
    n_reps = len(subjects[0][0])

    for s in range(n_subj):
        for t in range(n_tasks):
            task = tasks[s][t]
            task_features = compute_task_features(task)

            # Average FIM metrics across reps (same task, different behavioral noise)
            fim_metrics = []
            for r in range(n_reps):
                subj = subjects[s][t][r]
                fim_result = compute_task_fim(subj, task, noise_var)
                fim_metrics.append(fim_result)

            # Aggregate across reps
            row = {
                'subj_idx': s,
                'task_idx': t,
                'condition_number': np.mean([m['condition_number'] for m in fim_metrics]),
                'regressor_corr': np.mean([m['regressor_corr'] for m in fim_metrics]),
                'regressor_var_cpp': np.mean([m['regressor_var'][0] for m in fim_metrics]),
                'regressor_var_ru': np.mean([m['regressor_var'][1] for m in fim_metrics]),
                'fim_00': np.mean([m['fim'][0, 0] for m in fim_metrics]),
                'fim_11': np.mean([m['fim'][1, 1] for m in fim_metrics]),
                'fim_01': np.mean([m['fim'][0, 1] for m in fim_metrics]),
                'eig_max': np.mean([m['eigenvalues'][0] for m in fim_metrics]),
                'eig_min': np.mean([m['eigenvalues'][1] for m in fim_metrics]),
                'noise_var': np.mean([m['noise_var'] for m in fim_metrics]),
            }
            row.update(task_features)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Add error covariance eigenvalues (more interpretable than FIM eigenvalues)
    # Cov(beta_hat) = (n * FIM)^{-1}, so eig(Cov) = 1/(n * eig(FIM))
    df['err_var_min'] = 1.0 / (df['n_trials'] * df['eig_max'])  # min error variance
    df['err_var_max'] = 1.0 / (df['n_trials'] * df['eig_min'])  # max error variance
    df['err_sd_min'] = np.sqrt(df['err_var_min'])
    df['err_sd_max'] = np.sqrt(df['err_var_max'])

    return df
