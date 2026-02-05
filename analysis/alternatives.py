# Analysis functions for alternative cognitive models.
from configs import *
from changepoint.alternatives import (
    ALT_PARAM_RANGES, ALT_BLOCKS, N_ALT_SUBJECTS,
    simulate_alt_subjects, infer_beliefs
)
from changepoint.tasks import simulate_cpt
from analysis.analysis import fit_linear_models

LM_MODEL = 'model-pe-cpp-ru'

# Model display configuration: name -> (label, color)
ALT_MODEL_CONFIG = {
    'ph': ('Pearce-Hall', 'C0'),
    'li': ('Leaky Integrator', 'C1'),
    'real': ('Real data', 'C2'),
}


def get_result_stats(results_by_model, lm_model=LM_MODEL):
    """Compute summary statistics for each alternative model.

    Returns:
        dict of model_name -> {beta_*_mean, beta_*_sd, ve_mean, ve_sd}
    """
    summary = {}

    for model_name, result in results_by_model.items():
        lm = result['lm_results']

        beta_cols = [c for c in lm.columns if f'{lm_model}_beta_' in c]
        ve_col = f'{lm_model}_ve'

        stats = {}
        for col in beta_cols:
            short = col.replace(f'{lm_model}_', '')
            stats[f'{short}_mean'] = lm[col].mean()
            stats[f'{short}_sd'] = lm[col].std()

        stats['ve_mean'] = lm[ve_col].mean()
        stats['ve_sd'] = lm[ve_col].std()
        summary[model_name] = stats

    return summary


def print_result_stats(stats, lm_model=LM_MODEL, verbose=0):
    """Print summary table of betas and VE for each alternative model."""
    if verbose > 0: print(f"\nAlternative Model Beta Patterns ({lm_model}):")
    for model_name, model_stats in stats.items():
        if verbose > 0: print(f"\n  {model_name}:")
        for key, val in model_stats.items():
            if verbose > 0: print(f"    {key}: {val:.4f}")


def analyze_alternative_models(n_subjects=N_ALT_SUBJECTS, blocks=ALT_BLOCKS, lm_model=LM_MODEL, verbose=0):
    """Run alternative model analysis pipeline.

    Parameters:
        n_subjects (int) - Number of subjects to simulate per model.
        blocks (list)    - Block structure for simulated tasks.
        lm_model (str)   - Linear model to fit.
        verbose (int)    - Verbosity level (0=silent, 1=progress).

    Returns:
        dict with:
            'results'  - dict of model_name -> {'lm_results', 'true_params'}
            'tasks'    - list of simulated tasks
            'lm_model' - linear model used
    """
    if verbose > 0: print(f"Generating {n_subjects} tasks...")
    tasks = [simulate_cpt(blocks=blocks) for _ in range(n_subjects)]

    results = {}
    for model_name in ALT_PARAM_RANGES:
        if verbose > 0: print(f"\nRunning {model_name}:")
        if verbose > 0: print(f"  Simulating subjects...")
        subjects, true_params = simulate_alt_subjects(model_name, tasks)

        if verbose > 0: print(f"  Inferring Bayesian beliefs...")
        infer_beliefs(subjects, tasks)

        if verbose > 0: print(f"  Fitting linear models...")
        lm_results = fit_linear_models(subjects, model=lm_model)

        results[model_name] = {'lm_results': lm_results, 'true_params': true_params}

    stats = get_result_stats(results, lm_model=lm_model)
    print_result_stats(stats, lm_model=lm_model, verbose=verbose)

    return {'results': results, 'tasks': tasks, 'lm_model': lm_model}
