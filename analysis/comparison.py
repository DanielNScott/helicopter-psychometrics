# Model comparison analysis: recovery, variance explained, and reliability across linear models.
from configs import *
from analysis.analysis import fit_linear_models, get_model_terms, terms_to_params
from analysis.reliability import _get_split_half_reliabilities, _analysis_regression
from estimation.mle import parameter_recovery, save_recovery, load_recovery

# Models to compare: name -> (param_names, label, color)
COMPARISON_MODELS = {
    'model-pe-cpp-ru':             (terms_to_params(get_model_terms('model-pe-cpp-ru')),             'pe-cpp-ru',      'C0'),
    'model-pe-cpp-ru-prod':        (terms_to_params(get_model_terms('model-pe-cpp-ru-prod')),        'pe-cpp-ru-prod', 'C1'),
    'model-pe-cpp-ru-deltas':      (terms_to_params(get_model_terms('model-pe-cpp-ru-deltas')),      'deltas',         'C2'),
    'model-pe-cpp-ru-prod-deltas': (terms_to_params(get_model_terms('model-pe-cpp-ru-prod-deltas')), 'prod-deltas',    'C3'),
}

# Generative parameter ranges (same for all models)
GEN_PARAM_RANGES = {
    'beta_pe':  (0.0, 0.8),
    'beta_cpp': (0.0, 1.0),
    'beta_ru':  (0.0, 1.0),
}

# Block structure for recovery tasks
COMPARISON_BLOCKS = [
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
]


def compare_variance_explained(subjs, models=None):
    """Compute variance explained for each model.

    Parameters:
        subjs (list) - List of Subject objects with beliefs.
        models (dict) - Model name -> (param_names, label, color). Defaults to COMPARISON_MODELS.

    Returns:
        ve_results (dict) - Model name -> VE series (per subject).
    """
    if models is None:
        models = COMPARISON_MODELS

    ve_results = {}
    for model in models:
        lm = fit_linear_models(subjs, model=model)
        ve_results[model] = lm[f'{model}_ve']

    return ve_results


def compare_recovery(models=None, param_ranges=None, blocks=None, use_cache=True,
                     n_subjects=400, n_tasks_per_subject=5, n_reps=5, n_refits=1):
    """Run parameter recovery for each model.

    Parameters:
        models (dict)       - Model name -> (param_names, label, color). Defaults to COMPARISON_MODELS.
        param_ranges (dict) - Generative parameter ranges. Defaults to GEN_PARAM_RANGES.
        blocks (list)       - Block structure for tasks. Defaults to COMPARISON_BLOCKS.
        use_cache (bool)    - Whether to use cached results.
        n_subjects (int)    - Number of subjects to simulate.
        n_tasks_per_subject (int) - Tasks per subject.
        n_reps (int)        - Repetitions per subject-task.
        n_refits (int)      - Refits per estimate.

    Returns:
        recovery_results (dict) - Model name -> recovery result dict.
    """
    if models is None:
        models = COMPARISON_MODELS
    if param_ranges is None:
        param_ranges = GEN_PARAM_RANGES
    if blocks is None:
        blocks = COMPARISON_BLOCKS

    recovery_results = {}
    for model, (param_names, _, _) in models.items():

        # Try cache first
        cache_name = f'{model}_recovery'
        if use_cache:
            try:
                recovery_results[model] = load_recovery(cache_name)
                continue
            except FileNotFoundError:
                pass

        # Run recovery
        result = parameter_recovery(
            param_names=param_names,
            n_subjects=n_subjects,
            n_tasks_per_subject=n_tasks_per_subject,
            n_reps=n_reps,
            n_refits=n_refits,
            param_ranges=param_ranges,
            blocks=blocks,
            fit_method='ols',
            model=model,
        )
        save_recovery(result, cache_name)
        recovery_results[model] = result

    return recovery_results


def compare_reliability(subjs, tasks, models=None, nreps=20):
    """Compute split-half reliability for each model.

    Parameters:
        subjs (list)  - List of Subject objects.
        tasks (list)  - List of Task objects.
        models (dict) - Model name -> (param_names, label, color). Defaults to COMPARISON_MODELS.
        nreps (int)   - Number of split-half repetitions.

    Returns:
        reliability_results (dict) - Model name -> reliability DataFrame.
    """
    if models is None:
        models = COMPARISON_MODELS

    reliability_results = {}
    for model in models:

        # Build analysis function for this model
        def analysis_fn(subjs_a, subjs_b, tasks_a, tasks_b, m=model):
            return _analysis_regression(subjs_a, subjs_b, tasks_a, tasks_b, model=m)

        rel = _get_split_half_reliabilities(subjs, tasks, analysis_fn, nreps=nreps)
        reliability_results[model] = rel

    return reliability_results


def model_comparison_analysis(subjs, tasks, models=None, nreps=20):
    """Run full model comparison: variance explained, recovery, and reliability.

    Parameters:
        subjs (list)  - List of Subject objects with beliefs.
        tasks (list)  - List of Task objects.
        models (dict) - Model name -> (param_names, label, color). Defaults to COMPARISON_MODELS.
        nreps (int)   - Number of split-half repetitions.

    Returns:
        comparison (dict) with keys:
            'models'      - Model dict used
            'gen_params'  - Generative parameter names
            've'          - VE results by model
            'recovery'    - Recovery results by model
            'reliability' - Reliability results by model
    """
    if models is None:
        models = COMPARISON_MODELS

    ve_results = compare_variance_explained(subjs, models)
    recovery_results = compare_recovery(models, use_cache=USE_CACHE)
    reliability_results = compare_reliability(subjs, tasks, models, nreps=nreps)

    return {
        'models': models,
        'gen_params': list(GEN_PARAM_RANGES.keys()),
        've': ve_results,
        'recovery': recovery_results,
        'reliability': reliability_results,
    }
