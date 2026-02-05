# Systematic comparison of linear models on recovery, variance explained, and reliability.
from configs import *
from changepoint.dataio import read_experiment
from analysis.analysis import fit_linear_models, get_model_terms, terms_to_params
from analysis.reliability import _get_split_half_reliabilities, _analysis_regression
from estimation.mle import parameter_recovery, save_recovery, load_recovery

# Models to compare
MODELS = {
    'model-pe-cpp-ru':             terms_to_params(get_model_terms('model-pe-cpp-ru')),
    'model-pe-cpp-ru-prod':        terms_to_params(get_model_terms('model-pe-cpp-ru-prod')),
    'model-pe-cpp-ru-deltas':      terms_to_params(get_model_terms('model-pe-cpp-ru-deltas')),
    'model-pe-cpp-ru-prod-deltas': terms_to_params(get_model_terms('model-pe-cpp-ru-prod-deltas')),
}

# Generative parameter ranges (same for all models)
GEN_PARAM_RANGES = {
    'beta_pe':  (0.0, 0.8),
    'beta_cpp': (0.0, 1.0),
    'beta_ru':  (0.0, 1.0),
}

# Block structure for recovery tasks
BLOCKS = [
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
]


# ---- 1. Variance Explained ----

# Read experimental data
subjs, tasks = read_experiment(max_subj=MAX_SUBJ_NUM)

# Fit each model, collect VE
ve_results = {}
for model in MODELS:
    lm = fit_linear_models(subjs, model=model)
    ve_results[model] = lm[f'{model}_ve']

print("\nVariance Explained (mean across subjects):")
for model in MODELS:
    ve = ve_results[model]
    print(f"  {model}: {ve.mean():.4f} (sd={ve.std():.4f})")


# ---- 2. Recovery Simulations ----

recovery_results = {}
for model, param_names in MODELS.items():

    # Try cache first
    cache_name = f'{model}_recovery'
    if USE_CACHE:
        try:
            recovery_results[model] = load_recovery(cache_name)
            continue
        except FileNotFoundError:
            pass

    # Run recovery
    result = parameter_recovery(
        param_names=param_names,
        n_subjects=400,
        n_tasks_per_subject=5,
        n_reps=5,
        n_refits=1,
        param_ranges=GEN_PARAM_RANGES,
        blocks=BLOCKS,
        fit_method='ols',
        model=model,
    )
    save_recovery(result, cache_name)
    recovery_results[model] = result

# Cross-model recovery: correlate each model's betas with generative truth
gen_params = list(GEN_PARAM_RANGES.keys())

print("\nRecovery Correlations (model beta vs generative param):")
for model in MODELS:
    result = recovery_results[model]
    mean_ests = np.nanmean(result['results']['ests'], axis=(1, 2, 3))
    param_names = result['results']['param_names']

    print(f"\n  {model}:")
    for p, pname in enumerate(param_names):
        recovered = mean_ests[:, p]
        for gen_name in gen_params:
            true_vals = result['true_params'][gen_name]
            r = np.corrcoef(true_vals, recovered)[0, 1]
            print(f"    {pname} vs {gen_name}: r={r:.3f}")


# ---- 3. Split-Half Reliability ----

reliability_results = {}
for model in MODELS:

    # Build analysis function for this model
    def analysis_fn(subjs_a, subjs_b, tasks_a, tasks_b, m=model):
        return _analysis_regression(subjs_a, subjs_b, tasks_a, tasks_b, model=m)

    rel = _get_split_half_reliabilities(subjs, tasks, analysis_fn, nreps=20)
    reliability_results[model] = rel

print("\nSplit-Half Reliability (mean +/- sd):")
for model in MODELS:
    rel = reliability_results[model]
    print(f"\n  {model}:")
    for col in rel.columns:
        print(f"    {col}: {rel[col].mean():.3f} +/- {rel[col].std():.3f}")


# ---- 4. Figure 8: Model comparison plots ----

from plotting.figures import figure_8, compile_figure_8

figure_8(MODELS, recovery_results, reliability_results, ve_results, gen_params)
compile_figure_8()
