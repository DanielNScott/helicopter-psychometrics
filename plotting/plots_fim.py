from configs import *


def _fit_quadratic_surface(x, y, z):
    """
    Fit quadratic surface z = b0 + b1*x + b2*y + b3*x² + b4*xy + b5*y².

    Returns coefficients, R², and prediction function.
    """
    X = np.column_stack([
        np.ones_like(x),
        x, y,
        x**2, x*y, y**2
    ])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)

    # Compute R²
    z_pred = X @ coeffs
    ss_res = np.sum((z - z_pred)**2)
    ss_tot = np.sum((z - z.mean())**2)
    r2 = 1 - ss_res / ss_tot

    def predict(x_new, y_new):
        X_new = np.column_stack([
            np.ones_like(x_new.ravel()),
            x_new.ravel(), y_new.ravel(),
            x_new.ravel()**2, x_new.ravel()*y_new.ravel(), y_new.ravel()**2
        ])
        return (X_new @ coeffs).reshape(x_new.shape)

    return coeffs, r2, predict


def _kernel_smooth(x, y, z, x_grid, y_grid, bandwidth=None):
    """
    Gaussian kernel smoothing (Nadaraya-Watson estimator).

    If bandwidth is None, uses Scott's rule based on data spread.
    """
    if bandwidth is None:
        n = len(x)
        bandwidth = n**(-1/6) * (np.std(x) + np.std(y)) / 2

    z_grid = np.zeros_like(x_grid)

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            dx = (x - x_grid[i, j]) / bandwidth
            dy = (y - y_grid[i, j]) / bandwidth
            weights = np.exp(-0.5 * (dx**2 + dy**2))
            z_grid[i, j] = np.sum(weights * z) / np.sum(weights)

    return z_grid


def plot_task_scatter(fim_df, x_col, y_col, color_col=None, ax=None):
    """
    Plot scatter of two task features, optionally colored by a third.

    Parameters:
        fim_df (pd.DataFrame) - Output from analyze_task_information.
        x_col (str) - Column name for x-axis.
        y_col (str) - Column name for y-axis.
        color_col (str) - Column name for color. If None, uses default color.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.

    Returns:
        sc - Scatter object (for colorbar if needed).
    """
    if ax is None: fig, ax = plt.subplots()

    x = fim_df[x_col].values
    y = fim_df[y_col].values

    if color_col is not None:
        c = fim_df[color_col].values
        sc = ax.scatter(x, y, c=c, alpha=0.7, cmap='viridis')
        ax.set_title(color_col)
    else:
        r = np.corrcoef(x, y)[0, 1]
        sc = ax.scatter(x, y, alpha=0.5)
        ax.set_title(f'r = {r:.2f}')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)

    return sc


def plot_task_feature_correlations(fim_df, figsize=(10, 4)):
    """
    Plot correlations between task features to assess confounding structure.

    Two panels showing n_changepoints vs mean_run_length (expected near-functional)
    and n_changepoints vs std_run_length (expected weaker).

    Parameters:
        fim_df (pd.DataFrame) - Output from analyze_task_information.
        figsize (tuple) - Figure size.

    Returns:
        fig, axes - Figure and axes array.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_task_scatter(fim_df, 'n_changepoints', 'mean_run_length', ax=axes[0])
    plot_task_scatter(fim_df, 'n_changepoints', 'std_run_length', ax=axes[1])

    plt.tight_layout()
    return fig, axes


def plot_task_information(fim_df, figsize=(12, 4)):
    """
    Plot error metrics as function of task features.

    Three panels with n_changepoints (x) vs std_run_length (y), colored by
    condition_number, err_sd_max, and err_sd_min respectively.

    Parameters:
        fim_df (pd.DataFrame) - Output from analyze_task_information.
        figsize (tuple) - Figure size.

    Returns:
        fig, axes - Figure and axes array.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metrics = ['condition_number', 'err_sd_2', 'err_sd_0']

    for ax, metric in zip(axes, metrics):
        sc = plot_task_scatter(fim_df, 'n_changepoints', 'std_run_length', color_col=metric, ax=ax)
        fig.colorbar(sc, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig, axes



def plot_task_information_contours(fim_df, figsize=(12, 4), n_grid=50):
    """
    Plot quadratic fit contours for error metrics vs task features.

    Three panels with filled contours of quadratic surface fits.
    R² reported in titles.

    Parameters:
        fim_df (pd.DataFrame) - Output from analyze_task_information.
        figsize (tuple) - Figure size.
        n_grid (int) - Grid resolution for contours.

    Returns:
        fig, axes - Figure and axes array.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    x = fim_df['n_changepoints'].values
    y = fim_df['std_run_length'].values

    # Create grid for contour plotting
    x_lin = np.linspace(x.min(), x.max(), n_grid)
    y_lin = np.linspace(y.min(), y.max(), n_grid)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    metrics = ['condition_number', 'err_sd_2', 'err_sd_0']
    labels = ['Condition Number', 'Max Error SD', 'Min Error SD']

    for ax, metric, label in zip(axes, metrics, labels):
        z = fim_df[metric].values

        # Fit quadratic surface
        coeffs, r2, predict = _fit_quadratic_surface(x, y, z)
        z_grid = predict(x_grid, y_grid)

        # Plot filled contours with lines
        cf = ax.contourf(x_grid, y_grid, z_grid, levels=15, cmap='viridis', alpha=0.8)
        ax.contour(x_grid, y_grid, z_grid, levels=15, colors='k', linewidths=0.3, alpha=0.5)

        ax.set_xlabel('N Changepoints')
        ax.set_ylabel('Std Run Length')
        ax.set_title(f'{label} (R²={r2:.2f})')
        fig.colorbar(cf, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig, axes


def plot_task_information_kernel(fim_df, figsize=(12, 4), n_grid=50, bandwidth=None):
    """
    Plot kernel-smoothed contours for error metrics vs task features.

    Three panels with filled contours from Gaussian kernel smoothing.
    Bandwidth reported in titles.

    Parameters:
        fim_df (pd.DataFrame) - Output from analyze_task_information.
        figsize (tuple) - Figure size.
        n_grid (int) - Grid resolution for contours.
        bandwidth (float) - Kernel bandwidth. If None, uses Scott's rule.

    Returns:
        fig, axes - Figure and axes array.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    x = fim_df['n_changepoints'].values
    y = fim_df['std_run_length'].values

    # Create grid for contour plotting
    x_lin = np.linspace(x.min(), x.max(), n_grid)
    y_lin = np.linspace(y.min(), y.max(), n_grid)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    # Compute bandwidth once if not provided
    if bandwidth is None:
        n = len(x)
        bandwidth = n**(-1/6) * (np.std(x) + np.std(y)) / 2

    metrics = ['condition_number', 'err_sd_2', 'err_sd_0']
    labels = ['Condition Number', 'Max Error SD', 'Min Error SD']

    for ax, metric, label in zip(axes, metrics, labels):
        z = fim_df[metric].values

        # Kernel smooth
        z_grid = _kernel_smooth(x, y, z, x_grid, y_grid, bandwidth)

        # Plot filled contours with lines
        cf = ax.contourf(x_grid, y_grid, z_grid, levels=15, cmap='viridis', alpha=0.8)
        ax.contour(x_grid, y_grid, z_grid, levels=15, colors='k', linewidths=0.3, alpha=0.5)

        ax.set_xlabel('N Changepoints')
        ax.set_ylabel('Std Run Length')
        ax.set_title(f'{label} (bw={bandwidth:.2f})')
        fig.colorbar(cf, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig, axes
