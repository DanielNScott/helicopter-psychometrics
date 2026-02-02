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


# ---- 4. Plots ----

import os
os.makedirs(FIGURES_DIR, exist_ok=True)

# Short display names and colors for models
MODEL_LABELS = {
    'model-pe-cpp-ru':             'pe-cpp-ru',
    'model-pe-cpp-ru-prod':        'pe-cpp-ru-prod',
    'model-pe-cpp-ru-deltas':      'deltas',
    'model-pe-cpp-ru-prod-deltas': 'prod-deltas',
}
MODEL_COLORS = {
    'model-pe-cpp-ru':             'C0',
    'model-pe-cpp-ru-prod':        'C1',
    'model-pe-cpp-ru-deltas':      'C2',
    'model-pe-cpp-ru-prod-deltas': 'C3',
}


# --- Plot A: Recovery correlation by parameter and model ---

fig, ax = plt.subplots()
n_models = len(MODELS)
width = 0.8 / n_models

for m_idx, model in enumerate(MODELS):
    result = recovery_results[model]
    mean_ests = np.nanmean(result['results']['ests'], axis=(1, 2, 3))
    fit_param_names = result['results']['param_names']

    # Plot recovery r for each generative parameter
    for p_idx, gen_name in enumerate(gen_params):
        if gen_name not in fit_param_names:
            continue

        fit_idx = fit_param_names.index(gen_name)
        recovered = mean_ests[:, fit_idx]
        true_vals = result['true_params'][gen_name]
        r = np.corrcoef(true_vals, recovered)[0, 1]

        x = p_idx + (m_idx - (n_models - 1) / 2) * width
        ax.plot(x, r, 'o', color=MODEL_COLORS[model], markersize=7)

# Legend
for model in MODELS:
    ax.plot([], [], 'o', color=MODEL_COLORS[model], label=MODEL_LABELS[model])

ax.set_xticks(range(len(gen_params)))
ax.set_xticklabels(gen_params)
ax.set_ylabel('Recovery r (true vs recovered)')
ax.set_title('Parameter Recovery by Model')
ax.legend(fontsize=8)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR + 'supp_recovery_comparison' + FIG_FMT, dpi=300)


# --- Plot B: Split-half reliability by parameter and model ---

def _short_beta_name(col, model):
    """Extract short beta name from reliability column."""
    prefix = f'Rho {model}_'
    return col.replace(prefix, '') if col.startswith(prefix) else col

# Collect union of short param names across models, preserving order
all_short_names = []
for model in MODELS:
    rel = reliability_results[model]
    for col in rel.columns:
        short = _short_beta_name(col, model)
        if short not in all_short_names:
            all_short_names.append(short)

fig, ax = plt.subplots()
width = 0.8 / n_models

for m_idx, model in enumerate(MODELS):
    rel = reliability_results[model]

    for col in rel.columns:
        short = _short_beta_name(col, model)
        p_idx = all_short_names.index(short)
        mean_r = rel[col].mean()
        std_r = rel[col].std()

        x = p_idx + (m_idx - (n_models - 1) / 2) * width
        ax.errorbar(x, mean_r, yerr=std_r, fmt='o', color=MODEL_COLORS[model],
                    markersize=7, capsize=3)

# Legend
for model in MODELS:
    ax.plot([], [], 'o', color=MODEL_COLORS[model], label=MODEL_LABELS[model])

ax.set_xticks(range(len(all_short_names)))
ax.set_xticklabels(all_short_names, rotation=30, ha='right')
ax.set_ylabel('Split-half r (mean +/- sd)')
ax.set_title('Parameter Reliability by Model')
ax.legend(fontsize=8)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR + 'supp_reliability_comparison' + FIG_FMT, dpi=300)
