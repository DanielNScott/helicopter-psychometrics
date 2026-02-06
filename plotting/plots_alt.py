# Plotting functions for alternative cognitive models.
from configs import *
from analysis.alternatives import ALT_MODEL_CONFIG
from changepoint.alternatives import ALT_PARAM_RANGES, SIMULATE_FNS
from changepoint.subjects import get_beliefs


def plot_comparison_betas(results_by_model, real_lm, lm_model, ax=None):
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
        label, color = ALT_MODEL_CONFIG[source]

        # Get the linear model DataFrame for this source
        if source == 'real':
            lm = real_lm
        else:
            lm = results_by_model[source]['lm_results']

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

    return ax


def plot_comparison_ve(results_by_model, real_lm, lm_model, ax=None):
    """Plot variance explained distributions for each alternative model and real data."""
    if ax is None:
        fig, ax = plt.subplots()

    ve_col = f'{lm_model}_ve'

    # Collect all sources
    sources = list(results_by_model.keys()) + ['real']

    for s_idx, source in enumerate(sources):
        label, color = ALT_MODEL_CONFIG[source]

        if source == 'real':
            ve = real_lm[ve_col].values
        else:
            ve = results_by_model[source]['lm_results'][ve_col].values

        # Jittered strip
        xs = np.random.normal(s_idx, 0.08, len(ve))
        ax.scatter(xs, ve, c=color, alpha=0.3, s=10, zorder=2)
        ax.errorbar(s_idx, ve.mean(), yerr=ve.std(), fmt='D', color=color,
                    markersize=8, capsize=4, zorder=3, label=label)

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([ALT_MODEL_CONFIG[s][0] for s in sources])
    ax.set_ylabel('Variance Explained (R²)')
    ax.set_title('Linear Model VE by Generative Model')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return ax


def plot_example_subjects(results_by_model, tasks, lm_model, ntrials=120, ax_grid=None):
    """Plot example subject behavior for each alternative model.

    Parameters:
        results_by_model (dict) - Model name -> analysis result dict.
        tasks (list)            - List of simulated tasks.
        lm_model (str)          - Linear model used (for VE column).
        ntrials (int)           - Number of trials to show in traces.
        ax_grid (ndarray)       - Optional 2D array of axes (n_models x 3). Created if None.

    Returns:
        ax_grid (ndarray) - The axes with the plots.
    """
    model_names = list(results_by_model.keys())
    n_models = len(model_names)

    if ax_grid is None:
        fig, ax_grid = plt.subplots(n_models, 3, figsize=(16, 3.5 * n_models))
        if n_models == 1:
            ax_grid = ax_grid.reshape(1, -1)

    for row, model_name in enumerate(model_names):

        # Pick a median-VE subject as representative
        lm = results_by_model[model_name]['lm_results']
        ve = lm[f'{lm_model}_ve'].values
        snum = np.argsort(ve)[len(ve) // 2]

        # Re-simulate this one subject for access to beliefs
        param_ranges = ALT_PARAM_RANGES[model_name]
        true_params = results_by_model[model_name]['true_params']
        params = {name: true_params[name][snum] for name in param_ranges}
        subj = SIMULATE_FNS[model_name](tasks[snum].obs, **params)
        noise_sd = tasks[snum].noise_sd if not np.isscalar(tasks[snum].noise_sd) else np.full(len(tasks[snum].obs), tasks[snum].noise_sd)
        subj.beliefs = get_beliefs(subj.responses, tasks[snum].obs, tasks[snum].new_blk, tasks[snum].hazard, noise_sd)

        # Column 1: task observations and predictions
        ax = ax_grid[row, 0]
        ax.plot(tasks[snum].obs[:ntrials], '.', alpha=0.5, label='Obs')
        ax.plot(subj.responses.pred[:ntrials], '-', label='Pred')
        ax.set_ylabel('Value')
        ax.set_title(ALT_MODEL_CONFIG[model_name][0])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

        # Column 2: PE vs update scatter
        ax = ax_grid[row, 1]
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
        ax = ax_grid[row, 2]
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
    for ax in ax_grid[-1, :]:
        ax.set_xlabel('Trial')

    return ax_grid


def plot_model_learning_rates(task, ntrials=120, ax=None):
    """Plot learning rates for each model on the same task.

    Simulates each alternative model with median parameter values, plus the
    normative Bayesian model, and plots their learning rates over the first ntrials.

    Parameters:
        task (ChangepointTask) - Task to simulate on.
        ntrials (int)          - Number of trials to show.
        ax (Axes)              - Axes to plot on. If None, creates new figure.

    Returns:
        ax - The axes with the plot.
    """
    from changepoint.subjects import Subject, simulate_subject, DEFAULT_PARAMS_SUBJ

    if ax is None:
        fig, ax = plt.subplots()

    # Simulate normative Bayesian model
    norm_subj = Subject()
    norm_params = DEFAULT_PARAMS_SUBJ.copy()
    norm_params['noise_sd'] = task.noise_sd[0] if not np.isscalar(task.noise_sd) else task.noise_sd
    norm_params['hazard'] = task.hazard[0] if not np.isscalar(task.hazard) else task.hazard
    simulate_subject(norm_subj, task.obs, norm_params)
    ax.plot(norm_subj.responses.lr[:ntrials], label='Normative', color='black', alpha=0.8)

    # Simulate each alternative model with median parameters
    for model_name in ['ph', 'li']:
        param_ranges = ALT_PARAM_RANGES[model_name]
        sim_fn = SIMULATE_FNS[model_name]

        # Use median of parameter ranges
        params = {name: (lo + hi) / 2 for name, (lo, hi) in param_ranges.items()}
        subj = sim_fn(task.obs, **params)

        label, color = ALT_MODEL_CONFIG[model_name]
        ax.plot(subj.responses.lr[:ntrials], label=label, color=color, alpha=0.8)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Model Learning Rates on Same Task')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.5)

    return ax
