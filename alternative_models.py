# Simulate non-Bayesian cognitive models on the changepoint task and analyze
# their behavior through the Bayesian linear model pipeline.
from configs import *
from changepoint.subjects import Subject, Responses, get_beliefs, DEFAULT_PARAMS_SUBJ
from changepoint.tasks import simulate_cpt
from changepoint.dataio import read_experiment
from analysis.analysis import fit_linear_models
import os

# Alternative model parameter ranges
ALT_PARAM_RANGES = {
    'rw': {'alpha': (0.05, 0.95)},
    'ph': {'eta': (0.05, 0.95), 'alpha_0': (0.05, 0.95)},
    'li': {'lam': (0.05, 0.95), 'gamma': (0.0, 0.3)},
}

# Task blocks (same structure as supplement.py)
BLOCKS = [
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
]

N_SUBJECTS = 200
PRIOR_MEAN = 150.0
INIT_PRED = 150.0
PH_SCALE = 150.0
NOISE_SD_UPDATE = 0.5
LM_MODEL = 'model-pe-cpp-ru'


# ---- Simulation functions ----

def simulate_rw(obs, alpha, init_pred=INIT_PRED, noise_sd_update=NOISE_SD_UPDATE):
    """Rescorla-Wagner: fixed learning rate delta rule."""
    n = len(obs)
    responses = Responses(n)
    responses.pred[0] = init_pred

    for t in range(n):
        responses.pe[t] = obs[t] - responses.pred[t]
        noise = np.random.normal(0, noise_sd_update)
        responses.update[t] = alpha * responses.pe[t] + noise
        responses.pred[t + 1] = np.clip(responses.pred[t] + responses.update[t], 0, 300)
        responses.lr[t] = alpha

    return Subject(responses=responses)


def simulate_ph(obs, eta, alpha_0, init_pred=INIT_PRED, noise_sd_update=NOISE_SD_UPDATE,
                scale=PH_SCALE):
    """Pearce-Hall: attention-modulated learning rate that tracks surprise."""
    n = len(obs)
    responses = Responses(n)
    responses.pred[0] = init_pred
    alpha = alpha_0

    for t in range(n):
        responses.pe[t] = obs[t] - responses.pred[t]
        noise = np.random.normal(0, noise_sd_update)
        responses.update[t] = alpha * responses.pe[t] + noise
        responses.pred[t + 1] = np.clip(responses.pred[t] + responses.update[t], 0, 300)
        responses.lr[t] = alpha

        # Attention weight tracks recent surprise magnitude
        alpha = eta * np.abs(responses.pe[t]) / scale + (1 - eta) * alpha
        alpha = np.clip(alpha, 0.0, 1.0)

    return Subject(responses=responses)


def simulate_li(obs, lam, gamma, init_pred=INIT_PRED, noise_sd_update=NOISE_SD_UPDATE,
                prior_mean=PRIOR_MEAN):
    """Leaky integrator with prior pull: exponential decay plus mean reversion."""
    n = len(obs)
    responses = Responses(n)
    responses.pred[0] = init_pred

    for t in range(n):
        responses.pe[t] = obs[t] - responses.pred[t]
        prior_pull = gamma * (prior_mean - responses.pred[t])
        noise = np.random.normal(0, noise_sd_update)
        responses.update[t] = lam * responses.pe[t] + prior_pull + noise
        responses.pred[t + 1] = np.clip(responses.pred[t] + responses.update[t], 0, 300)

        # Effective learning rate
        if responses.pe[t] != 0:
            responses.lr[t] = responses.update[t] / responses.pe[t]
        else:
            responses.lr[t] = lam

    return Subject(responses=responses)


# Dispatch table for simulation functions
SIMULATE_FNS = {
    'rw': simulate_rw,
    'ph': simulate_ph,
    'li': simulate_li,
}


# ---- Batch simulation and analysis ----

def simulate_alt_subjects(model_name, tasks):
    """Simulate many subjects using an alternative model with varied parameters.

    Parameters:
        model_name (str) - 'rw', 'ph', or 'li'
        tasks (list)     - List of ChangepointTask objects, one per subject

    Returns:
        subjects (list)     - List of Subject instances with .responses populated
        true_params (dict)  - {param_name: array of true values}
    """
    param_ranges = ALT_PARAM_RANGES[model_name]
    sim_fn = SIMULATE_FNS[model_name]
    n = len(tasks)

    # Sample parameters uniformly for each subject
    true_params = {name: np.random.uniform(lo, hi, n) for name, (lo, hi) in param_ranges.items()}

    # Simulate each subject
    subjects = []
    for i in range(n):
        params = {name: true_params[name][i] for name in param_ranges}
        subj = sim_fn(tasks[i].obs, **params)
        subjects.append(subj)

    return subjects, true_params


def infer_beliefs(subjects, tasks):
    """Infer Bayesian beliefs (CPP, RU) from alternative model responses. Modifies subjects in-place."""
    for subj, task in zip(subjects, tasks):
        noise_sd = task.noise_sd if not np.isscalar(task.noise_sd) else np.full(len(task.obs), task.noise_sd)
        subj.beliefs = get_beliefs(subj.responses, task.obs, task.new_blk, task.hazard, noise_sd)


def analyze_alt_model(model_name, tasks, lm_model=LM_MODEL):
    """Run full analysis pipeline for one alternative model.

    Generates subjects, infers Bayesian beliefs, fits linear models.

    Returns:
        dict with 'lm_results' (DataFrame), 'true_params' (dict), 'model_name' (str)
    """
    print(f"  Simulating {model_name} subjects...")
    subjects, true_params = simulate_alt_subjects(model_name, tasks)

    print(f"  Inferring Bayesian beliefs...")
    infer_beliefs(subjects, tasks)

    print(f"  Fitting linear models...")
    lm_results = fit_linear_models(subjects, model=lm_model)

    return {'lm_results': lm_results, 'true_params': true_params, 'model_name': model_name}


def summarize_results(results_by_model, lm_model=LM_MODEL):
    """Print summary table of betas and VE for each alternative model."""
    summary = {}

    for model_name, result in results_by_model.items():
        lm = result['lm_results']

        # Collect beta and VE columns
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

    # Print table
    print(f"\nAlternative Model Beta Patterns ({lm_model}):")
    for model_name, stats in summary.items():
        print(f"\n  {model_name}:")
        for key, val in stats.items():
            print(f"    {key}: {val:.4f}")

    return summary


# ---- Plot functions ----

MODEL_LABELS = {'rw': 'Rescorla-Wagner', 'ph': 'Pearce-Hall', 'li': 'Leaky Integrator'}
MODEL_COLORS = {'rw': 'C0', 'ph': 'C1', 'li': 'C2', 'real': 'C3'}


def plot_beta_comparison(results_by_model, real_lm, lm_model=LM_MODEL, ax=None):
    """Plot mean betas (+/- SD) for each alternative model and real data side by side."""
    if ax is None:
        fig, ax = plt.subplots()

    # Get beta term names (skip intercept 'c')
    beta_cols = [c for c in real_lm.columns if f'{lm_model}_beta_' in c and not c.endswith('_c')]
    term_labels = [c.split('_beta_')[-1].upper() for c in beta_cols]

    # Collect all sources: alternative models + real data
    sources = list(results_by_model.keys()) + ['real']
    n_sources = len(sources)
    width = 0.8 / n_sources

    for s_idx, source in enumerate(sources):

        # Get the linear model DataFrame for this source
        if source == 'real':
            lm = real_lm
            color = MODEL_COLORS['real']
            label = 'Real data'
        else:
            lm = results_by_model[source]['lm_results']
            color = MODEL_COLORS[source]
            label = MODEL_LABELS[source]

        # Plot mean +/- SD for each beta
        for p_idx, col in enumerate(beta_cols):
            x = p_idx + (s_idx - (n_sources - 1) / 2) * width
            mean = lm[col].mean()
            sd = lm[col].std()
            marker_kw = dict(fmt='o', color=color, markersize=6, capsize=3)
            if p_idx == 0:
                ax.errorbar(x, mean, yerr=sd, label=label, **marker_kw)
            else:
                ax.errorbar(x, mean, yerr=sd, **marker_kw)

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xticks(range(len(term_labels)))
    ax.set_xticklabels(term_labels)
    ax.set_ylabel('Beta (mean +/- SD)')
    ax.set_title('Linear Model Betas by Generative Model')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_ve_comparison(results_by_model, real_lm, lm_model=LM_MODEL, ax=None):
    """Plot variance explained distributions for each alternative model and real data."""
    if ax is None:
        fig, ax = plt.subplots()

    ve_col = f'{lm_model}_ve'

    # Collect all sources
    sources = list(results_by_model.keys()) + ['real']

    for s_idx, source in enumerate(sources):
        if source == 'real':
            ve = real_lm[ve_col].values
            color = MODEL_COLORS['real']
            label = 'Real data'
        else:
            ve = results_by_model[source]['lm_results'][ve_col].values
            color = MODEL_COLORS[source]
            label = MODEL_LABELS[source]

        # Jittered strip
        xs = np.random.normal(s_idx, 0.08, len(ve))
        ax.scatter(xs, ve, c=color, alpha=0.3, s=10, zorder=2)
        ax.errorbar(s_idx, ve.mean(), yerr=ve.std(), fmt='D', color=color,
                    markersize=8, capsize=4, zorder=3, label=label)

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([MODEL_LABELS.get(s, 'Real data') for s in sources])
    ax.set_ylabel('Variance Explained (R²)')
    ax.set_title('Linear Model VE by Generative Model')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_example_subjects(results_by_model, tasks, ntrials=120, savefig=True):
    """Plot example subject behavior for each alternative model (3 rows x 3 cols).

    Columns: task + predictions, PE vs update scatter, CPP/RU/LR traces.
    Rows: one per model (RW, PH, LI).
    """
    model_names = list(results_by_model.keys())
    fig, axes = plt.subplots(len(model_names), 3, figsize=(16, 3.5 * len(model_names)))

    for row, model_name in enumerate(model_names):

        # Pick a median-VE subject as representative
        lm = results_by_model[model_name]['lm_results']
        ve = lm[f'{LM_MODEL}_ve'].values
        snum = np.argsort(ve)[len(ve) // 2]

        # Re-simulate this one subject for access to beliefs
        param_ranges = ALT_PARAM_RANGES[model_name]
        true_params = results_by_model[model_name]['true_params']
        params = {name: true_params[name][snum] for name in param_ranges}
        subj = SIMULATE_FNS[model_name](tasks[snum].obs, **params)
        noise_sd = tasks[snum].noise_sd if not np.isscalar(tasks[snum].noise_sd) else np.full(len(tasks[snum].obs), tasks[snum].noise_sd)
        subj.beliefs = get_beliefs(subj.responses, tasks[snum].obs, tasks[snum].new_blk, tasks[snum].hazard, noise_sd)

        # Column 1: task observations and predictions
        ax = axes[row, 0]
        ax.plot(tasks[snum].obs[:ntrials], '.', alpha=0.5, label='Obs')
        ax.plot(subj.responses.pred[:ntrials], '-', label='Pred')
        ax.set_ylabel('Value')
        ax.set_title(MODEL_LABELS[model_name])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

        # Column 2: PE vs update scatter
        ax = axes[row, 1]
        ax.scatter(subj.responses.pe, subj.responses.update, alpha=0.3, s=10)
        lims = [-150, 150]
        ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlabel('PE')
        ax.set_ylabel('Update')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title('PE vs Update')
        ax.grid(alpha=0.3)

        # Column 3: CPP, RU, LR traces
        ax = axes[row, 2]
        ax.plot(subj.beliefs.cpp[:ntrials], label='CPP')
        ax.plot(subj.beliefs.relunc[:ntrials], label='RU')
        clipped_lr = np.clip(subj.responses.lr[:ntrials], -1, 2)
        ax.plot(clipped_lr, 'k--', alpha=0.5, label='LR')
        ax.set_ylim(-0.2, 1.5)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Value')
        ax.set_title('Inferred CPP / RU / Model LR')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    # Label bottom row x-axes
    for ax in axes[-1, :]:
        ax.set_xlabel('Trial')

    fig.tight_layout()

    if savefig:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(FIGURES_DIR + 'alt_models_examples' + FIG_FMT, dpi=300)


def plot_alt_models(results_by_model, real_lm, tasks, lm_model=LM_MODEL, savefig=True, close=True):
    """Generate all alternative model figures."""
    if savefig:
        os.makedirs(FIGURES_DIR, exist_ok=True)

    # Panel A: beta comparison
    fig, ax = plt.subplots()
    plot_beta_comparison(results_by_model, real_lm, lm_model, ax=ax)
    fig.tight_layout()
    if savefig:
        fig.savefig(FIGURES_DIR + 'alt_models_betas' + FIG_FMT, dpi=300)

    # Panel B: VE comparison
    fig, ax = plt.subplots()
    plot_ve_comparison(results_by_model, real_lm, lm_model, ax=ax)
    fig.tight_layout()
    if savefig:
        fig.savefig(FIGURES_DIR + 'alt_models_ve' + FIG_FMT, dpi=300)

    # Panel C: example subject traces
    plot_example_subjects(results_by_model, tasks, savefig=savefig)

    if close:
        plt.close('all')


# ---- Main script ----

# Generate shared tasks
print(f"Generating {N_SUBJECTS} tasks...")
tasks = [simulate_cpt(blocks=BLOCKS) for _ in range(N_SUBJECTS)]

# Run each alternative model
results = {}
for model_name in ALT_PARAM_RANGES:
    print(f"\nRunning {model_name}:")
    results[model_name] = analyze_alt_model(model_name, tasks)

# Summary of alternative models
summary = summarize_results(results)

# Compare with real data
subjs, _ = read_experiment(max_subj=MAX_SUBJ_NUM)
real_lm = fit_linear_models(subjs, model=LM_MODEL)

print(f"\nReal data ({LM_MODEL}):")
for col in [c for c in real_lm.columns if 'beta_' in c]:
    short = col.replace(f'{LM_MODEL}_', '')
    print(f"  {short}_mean: {real_lm[col].mean():.4f}")
    print(f"  {short}_sd: {real_lm[col].std():.4f}")
print(f"  ve_mean: {real_lm[f'{LM_MODEL}_ve'].mean():.4f}")
print(f"  ve_sd: {real_lm[f'{LM_MODEL}_ve'].std():.4f}")

# ---- 5. Figures ----
plot_alt_models(results, real_lm, tasks)
