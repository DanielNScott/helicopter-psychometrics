# Plotting functions for model comparison analysis.
from configs import *


def plot_recovery_by_model(models, recovery_results, gen_params, ax=None):
    """Plot recovery correlation by parameter and model.

    Parameters:
        models (dict)           - Model name -> (param_names, label, color).
        recovery_results (dict) - Model name -> recovery result dict.
        gen_params (list)       - Generative parameter names.
        ax (Axes)               - Matplotlib axes. Created if None.

    Returns:
        ax (Axes) - The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    n_models = len(models)
    width = 0.8 / n_models

    # Plot recovery correlations
    for m_idx, (model, (_, label, color)) in enumerate(models.items()):
        result = recovery_results[model]
        mean_ests = np.nanmean(result['results']['ests'], axis=(1, 2, 3))
        fit_param_names = result['results']['param_names']

        for p_idx, gen_name in enumerate(gen_params):
            if gen_name not in fit_param_names:
                continue
            fit_idx = fit_param_names.index(gen_name)
            recovered = mean_ests[:, fit_idx]
            true_vals = result['true_params'][gen_name]
            r = np.corrcoef(true_vals, recovered)[0, 1]
            x = p_idx + (m_idx - (n_models - 1) / 2) * width
            ax.plot(x, r, 'o', color=color, markersize=7)

    # Add legend entries
    for model, (_, label, color) in models.items():
        ax.plot([], [], 'o', color=color, label=label)

    ax.set_xticks(range(len(gen_params)))
    ax.set_xticklabels(gen_params)
    ax.set_ylabel('Recovery r (true vs recovered)')
    ax.set_title('Parameter Recovery by Model')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    return ax


def plot_reliability_by_model(models, reliability_results, ax=None):
    """Plot split-half reliability by parameter and model.

    Parameters:
        models (dict)              - Model name -> (param_names, label, color).
        reliability_results (dict) - Model name -> reliability DataFrame.
        ax (Axes)                  - Matplotlib axes. Created if None.

    Returns:
        ax (Axes) - The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    n_models = len(models)
    width = 0.8 / n_models

    def _short_beta_name(col, model):
        prefix = f'Rho {model}_'
        return col.replace(prefix, '') if col.startswith(prefix) else col

    # Collect all unique short names across models
    all_short_names = []
    for model in models:
        rel = reliability_results[model]
        for col in rel.columns:
            short = _short_beta_name(col, model)
            if short not in all_short_names:
                all_short_names.append(short)

    # Plot reliability values
    for m_idx, (model, (_, label, color)) in enumerate(models.items()):
        rel = reliability_results[model]
        for col in rel.columns:
            short = _short_beta_name(col, model)
            p_idx = all_short_names.index(short)
            mean_r = rel[col].mean()
            std_r = rel[col].std()
            x = p_idx + (m_idx - (n_models - 1) / 2) * width
            ax.errorbar(x, mean_r, yerr=std_r, fmt='o', color=color,
                        markersize=7, capsize=3)

    # Add legend entries
    for model, (_, label, color) in models.items():
        ax.plot([], [], 'o', color=color, label=label)

    ax.set_xticks(range(len(all_short_names)))
    ax.set_xticklabels(all_short_names, rotation=30, ha='right')
    ax.set_ylabel('Split-half r (mean +/- sd)')
    ax.set_title('Parameter Reliability by Model')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    return ax


def plot_ve_by_model(models, ve_results, ax=None):
    """Plot variance explained by model.

    Parameters:
        models (dict)     - Model name -> (param_names, label, color).
        ve_results (dict) - Model name -> VE array (per subject).
        ax (Axes)         - Matplotlib axes. Created if None.

    Returns:
        ax (Axes) - The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    positions = range(len(models))

    # Plot VE for each model
    for m_idx, (model, (_, label, color)) in enumerate(models.items()):
        ve = ve_results[model]
        mean_ve = ve.mean()
        std_ve = ve.std()
        ax.errorbar(m_idx, mean_ve, yerr=std_ve, fmt='o', color=color,
                    markersize=7, capsize=3)

    # Set x-axis labels
    labels = [info[1] for info in models.values()]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Variance explained (R²)')
    ax.set_title('Model Variance Explained')
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    return ax
