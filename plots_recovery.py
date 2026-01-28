from configs import *

def _get_2d_colors(x, y):
    """Compute RGB colors from 2D coordinates using red-green gradient.

    Maps normalized x to red channel, normalized y to green channel.
    """
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    colors = np.column_stack([x_norm, y_norm, np.full_like(x_norm, 0.3)])
    return colors


def plot_param_recovery(analysis, param_name, colors=None, ax=None):
    """Plot true vs recovered parameter values with identity line and correlation.

    Parameters:
        analysis (dict) - Output from analyze_recovery, keyed by parameter name.
        param_name (str) - Parameter to plot (e.g., 'beta_cpp', 'beta_ru').
        colors (array) - RGB colors for each point. If None, uses default.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    a = analysis[param_name]
    true_vals = a['true_values']
    rec_vals = a['recovered_values']

    # Scatter plot
    ax.scatter(true_vals, rec_vals, c=colors, alpha=0.6)

    # Identity line
    lims = [min(true_vals.min(), rec_vals.min()), max(true_vals.max(), rec_vals.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)

    # Labels and title with correlation
    ax.set_xlabel(f'True {param_name}')
    ax.set_ylabel(f'Recovered {param_name}')
    ax.set_title(f'{param_name}: r = {a["correlation"]:.3f}')
    ax.grid(alpha=0.3)


def plot_param_scatter(analysis, param_x, param_y, colors=None, recovered=False, ax=None):
    """Plot scatter of two parameters (true or recovered values).

    Parameters:
        analysis (dict) - Output from analyze_recovery, keyed by parameter name.
        param_x (str) - Parameter for x-axis.
        param_y (str) - Parameter for y-axis.
        colors (array) - RGB colors for each point. If None, uses default.
        recovered (bool) - If True, plot recovered values; else plot true values.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    key = 'recovered_values' if recovered else 'true_values'
    x_vals = analysis[param_x][key]
    y_vals = analysis[param_y][key]

    ax.scatter(x_vals, y_vals, c=colors, alpha=0.6)

    label = 'Recovered' if recovered else 'True'
    ax.set_xlabel(f'{label} {param_x}')
    ax.set_ylabel(f'{label} {param_y}')
    ax.set_title(f'{label} Parameter Values')
    ax.grid(alpha=0.3)


def plot_recovery_summary(analysis, param_names=['beta_cpp', 'beta_ru'], figsize=(10, 8)):
    """Plot 2x2 grid of parameter recovery diagnostics.

    Top row: true vs recovered for each parameter.
    Bottom row: true values scatter, recovered values scatter.
    Points colored by 2D gradient based on true parameter values.

    Parameters:
        analysis (dict) - Output from analyze_recovery.
        param_names (list) - Two parameter names to plot.
        figsize (tuple) - Figure size.

    Returns:
        fig, axes - Figure and axes array.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Compute 2D color gradient from true parameter values
    true_x = analysis[param_names[0]]['true_values']
    true_y = analysis[param_names[1]]['true_values']
    colors = _get_2d_colors(true_x, true_y)

    # Top row: true vs recovered
    plot_param_recovery(analysis, param_names[0], colors=colors, ax=axes[0, 0])
    plot_param_recovery(analysis, param_names[1], colors=colors, ax=axes[0, 1])

    # Bottom row: parameter scatter plots
    plot_param_scatter(analysis, param_names[0], param_names[1], colors=colors, recovered=False, ax=axes[1, 0])
    plot_param_scatter(analysis, param_names[0], param_names[1], colors=colors, recovered=True, ax=axes[1, 1])

    plt.tight_layout()
    return fig, axes


def plot_error_corr_matrix(err_analysis, ax=None):
    """Plot error correlation matrix.

    Parameters:
        err_analysis (dict) - Output from analyze_error_covariance.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.

    Returns:
        im - Image object for colorbar if needed.
    """
    if ax is None: fig, ax = plt.subplots()

    param_names = err_analysis['param_names']
    n_params = len(param_names)

    im = ax.imshow(err_analysis['corr'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    ax.set_title('Error Correlation')

    return im


def plot_error_eigenvalues(err_analysis, ax=None):
    """Plot error covariance eigenvalue spectrum.

    Parameters:
        err_analysis (dict) - Output from analyze_error_covariance.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    param_names = err_analysis['param_names']
    n_params = len(param_names)
    eigs = err_analysis['eigenvalues']

    ax.bar(range(n_params), eigs)
    ax.set_xticks(range(n_params))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_params)])
    ax.set_ylabel('Eigenvalue (error variance)')
    ax.set_title('Error Covariance Spectrum')


def plot_error_eigenvectors(err_analysis, ax=None):
    """Plot error covariance eigenvector loadings.

    Parameters:
        err_analysis (dict) - Output from analyze_error_covariance.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    param_names = err_analysis['param_names']
    n_params = len(param_names)
    evecs = err_analysis['eigenvectors']

    x = np.arange(n_params)
    width = 0.8 / n_params
    for i in range(n_params):
        ax.bar(x + i * width, evecs[:, i], width, label=f'PC{i+1}')
    ax.set_xticks(x + width * (n_params - 1) / 2)
    ax.set_xticklabels(param_names)
    ax.set_ylabel('Loading')
    ax.set_title('Eigenvector Loadings')
    ax.legend()
    ax.axhline(0, color='k', linewidth=0.5)


def plot_error_covariance(err_analysis, figsize=(12, 4)):
    """Plot estimation error covariance analysis.

    Three panels: error correlation matrix, eigenvalue spectrum, eigenvector loadings.

    Parameters:
        err_analysis (dict) - Output from analyze_error_covariance.
        figsize (tuple) - Figure size.

    Returns:
        fig, axes - Figure and axes array.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im = plot_error_corr_matrix(err_analysis, ax=axes[0])
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    plot_error_eigenvalues(err_analysis, ax=axes[1])
    plot_error_eigenvectors(err_analysis, ax=axes[2])

    plt.tight_layout()
    return fig, axes


def plot_fim_analysis(fim_analysis, err_analysis=None, figsize=(10, 4)):
    """Plot Fisher information analysis with optional comparison to empirical errors.

    Two or three panels: FIM structure, FIM vs inverse-FIM eigenvalues,
    and optionally comparison of FIM-predicted vs empirical error covariance.

    Parameters:
        fim_analysis (dict) - Output from compute_regressor_fim.
        err_analysis (dict) - Optional output from analyze_error_covariance.
        figsize (tuple) - Figure size.

    Returns:
        fig, axes - Figure and axes array.
    """
    n_panels = 3 if err_analysis is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    param_names = fim_analysis['param_names']

    # Panel 1: FIM as heatmap
    fim = fim_analysis['fim']
    im = axes[0].imshow(fim, cmap='viridis')
    axes[0].set_xticks(range(len(param_names)))
    axes[0].set_yticks(range(len(param_names)))
    axes[0].set_xticklabels(param_names, rotation=45, ha='right')
    axes[0].set_yticklabels(param_names)
    axes[0].set_title(f'Per-Trial FIM (cond={fim_analysis["condition_number"]:.1f})')
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # Panel 2: eigenvalue comparison (FIM vs inverse)
    eigs_fim = fim_analysis['eigenvalues']
    eigs_inv = 1.0 / eigs_fim
    x = np.arange(len(param_names))
    width = 0.35
    axes[1].bar(x - width/2, eigs_fim / eigs_fim.max(), width, label='FIM')
    axes[1].bar(x + width/2, eigs_inv / eigs_inv.max(), width, label='FIM$^{-1}$')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'PC{i+1}' for i in range(len(param_names))])
    axes[1].set_ylabel('Normalized Eigenvalue')
    axes[1].set_title('Information vs Uncertainty')
    axes[1].legend()

    # Panel 3: compare predicted vs empirical covariance (if available)
    if err_analysis is not None:
        predicted_cov = fim_analysis['fim_inv']
        empirical_cov = err_analysis['cov']

        # Scale predicted to match empirical (FIM gives shape, not absolute scale)
        scale = np.trace(empirical_cov) / np.trace(predicted_cov)
        predicted_scaled = predicted_cov * scale

        # Plot diagonal and off-diagonal elements
        labels = ['Var(' + p + ')' for p in param_names] + ['Cov']
        pred_vals = [predicted_scaled[0, 0], predicted_scaled[1, 1], predicted_scaled[0, 1]]
        emp_vals = [empirical_cov[0, 0], empirical_cov[1, 1], empirical_cov[0, 1]]

        x = np.arange(len(labels))
        axes[2].bar(x - width/2, pred_vals, width, label='FIM predicted')
        axes[2].bar(x + width/2, emp_vals, width, label='Empirical')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)
        axes[2].set_ylabel('Covariance')
        axes[2].set_title('Predicted vs Empirical Error Cov')
        axes[2].legend()
        axes[2].axhline(0, color='k', linewidth=0.5)

    plt.tight_layout()
    return fig, axes


def plot_variance_decomposition(analysis, ax=None):
    """Plot recovery error breakdown by source (task vs repetition variability).

    Shows coefficient of variation (std/mean) for task and rep sources.

    Parameters:
        analysis (dict) - Output from analyze_recovery, keyed by parameter name.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    param_names = list(analysis.keys())
    n_params = len(param_names)

    # Compute coefficient of variation for each source
    cv_task = []
    cv_rep = []
    for name in param_names:
        a = analysis[name]
        mean_est = np.nanmean(a['recovered_values'])
        std_task = np.sqrt(a['var_task'])
        std_rep = np.sqrt(a['var_rep'])
        cv_task.append(std_task / np.abs(mean_est) if mean_est != 0 else 0)
        cv_rep.append(std_rep / np.abs(mean_est) if mean_est != 0 else 0)

    x = np.arange(n_params)
    width = 0.35

    ax.bar(x - width/2, cv_task, width, label='Task variability')
    ax.bar(x + width/2, cv_rep, width, label='Rep variability')

    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.set_ylabel('CV (std / mean)')
    ax.set_title('Error Source Decomposition')
    ax.legend()
    ax.grid(alpha=0.3)
