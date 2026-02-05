"""Plot functions for helicopter psychometrics figures."""

from configs import *


def plot_task_example(subjs, tasks, snum=0, ax=None):
    """Plot task observations and subject predictions for a single subject."""
    if ax is None: fig, ax = plt.subplots()

    ntrials = 120
    ax.plot(subjs[snum].responses.pred[0:ntrials], '-', label='Pred')
    ax.plot(tasks[snum].obs[0:ntrials], '.', label='Obs')
    ax.legend(loc='lower right')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Obs./Pred.')


def plot_subject_update_by_pe(subj, ax=None):
    """Plot prediction error vs update scatter for a single subject."""
    if ax is None: fig, ax = plt.subplots()

    ax.plot(subj.responses.pe, subj.responses.update, '.')
    ax.set_xlabel('Prediction error')
    ax.set_ylabel('Update')


def plot_peri_cp_lr_bars(subj_pcp_lr, snum=0, ax=None):
    """Plot peri-changepoint learning rates as a bar chart for a single subject.

    Parameters:
        subj_pcp_lr (pd.DataFrame) - DataFrame with peri-CP learning rates, subjects as rows.
        snum (int) - Subject index to plot.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()


    # Get column labels for x-axis
    col_labels = subj_pcp_lr.columns.tolist()
    xs = range(len(col_labels))

    # Plot bars
    ax.bar(xs, subj_pcp_lr.iloc[snum].values)
    ax.set_xticks(xs)
    ax.set_xticklabels(col_labels)
    ax.set_xlabel('Trial Relative to CP')
    ax.set_ylabel('Learning Rate')


def plot_pe_update_model_comparison(subj_pe, subj_cpp, ax=None):
    """Plot PE vs Update scatter comparing PE-only and CPP-based models.

    Parameters:
        subj_pe (Subject) - Subject with PE-only model (constant learning rate).
        subj_cpp (Subject) - Subject with CPP-based model (adaptive learning rate).
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    # Plot PE-only model (constant LR = diagonal)
    ax.plot(subj_pe.responses.pe, subj_pe.responses.update, '.', alpha=0.5, label='PE Only')

    # Plot CPP-based model (adaptive LR = varied slope)
    ax.plot(subj_cpp.responses.pe, subj_cpp.responses.update, '.', alpha=0.5, label='CPP-Based')

    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Update')
    ax.legend()


def plot_subj_betas_strip(betas, model='model-pe-cpp-ru-prod-deltas', ax=None):
    """Plot strip plot of subject beta coefficients with normative reference.

    Parameters:
        betas (pd.DataFrame) - DataFrame with beta columns from fit_linear_models.
        model (str) - Model prefix for column names.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    # Auto-detect beta columns for this model
    cols = [c for c in betas.columns if c.startswith(f'{model}_beta_')]
    labels = [c.split('_beta_')[-1].upper() for c in cols]

    nsubj = len(betas)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot each beta as a strip, normalizing intercept by 1/300
    for i, col in enumerate(cols):
        xs = np.random.rand(nsubj) * 0.3 - 0.15 + i
        ys = betas[col].values
        if col.endswith('_beta_c'):
            ys = ys / 300
        ax.scatter(xs, ys, c=colors[i], alpha=0.6, zorder=2)

    # Plot means, normalizing intercept by 1/300
    means = []
    for col in cols:
        m = betas[col].mean()
        if col.endswith('_beta_c'):
            m = m / 300
        means.append(m)
    ax.scatter(range(len(cols)), means, c='k', marker='x', s=80, zorder=3, label='Mean')

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Beta Coefficient')
    ax.grid(alpha=0.3, zorder=-10)


def plot_lm_ve(subjs, ax=None):
    """Plot trialwise regression variance explained for nested models.

    Parameters:
        subjs (list) - List of Subject objects with beliefs.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    from analysis.analysis import fit_linear_models

    models = ['model-pe', 'model-pe-cpp', 'model-pe-cpp-ru']
    labels = ['pe', 'pe-cpp', 'pe-cpp-ru']

    # Get variance explained by each model
    ve_1 = fit_linear_models(subjs, model=models[0])[f'{models[0]}_ve'].values
    ve_2 = fit_linear_models(subjs, model=models[1])[f'{models[1]}_ve'].values
    ve_3 = fit_linear_models(subjs, model=models[2])[f'{models[2]}_ve'].values

    # Sort on the full model's variance explained
    inds = np.argsort(ve_3)

    ax.plot(ve_1[inds], '.', alpha=0.7, label=labels[0])
    ax.plot(ve_2[inds], '.', alpha=0.7, label=labels[1])
    ax.plot(ve_3[inds], '.', alpha=0.7, label=labels[2])
    ax.set_xlabel('Sorted Subject Number')
    ax.set_ylabel('Variance Explained')
    ax.set_title('Trialwise Regression VE')
    ax.legend()


def plot_cumulative_ve(cumulative_ve_pca, cumulative_ve_lm=None, ax=None):
    """Plot cumulative variance explained by PCA and optionally linear models.

    Parameters:
        cumulative_ve_pca (array) - Cumulative variance explained by PCA.
        cumulative_ve_lm (array) - Cumulative variance explained by linear models.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    # Limit to first 3 components
    pca_vals = cumulative_ve_pca[:3]
    ax.plot(range(len(pca_vals)), pca_vals, '-o', label='PCA')

    if cumulative_ve_lm is not None:
        lm_vals = cumulative_ve_lm[:3]
        ax.plot(range(len(lm_vals)), lm_vals, '-s', label='Linear Model')
        ax.legend()

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Cumulative Variance Explained')


def plot_scores_strip(scores, ax=None):
    """Plot strip plot of subject PCA scores.

    Parameters:
        scores (pd.DataFrame) - DataFrame with Score_0, Score_1, etc. columns.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    # Get score columns
    cols = [c for c in scores.columns if c.startswith('Score_')]
    labels = [f'PC{i+1}' for i in range(len(cols))]

    nsubj = len(scores)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot each score as a strip
    for i, col in enumerate(cols):
        xs = np.random.rand(nsubj) * 0.3 - 0.15 + i
        ys = scores[col].values
        ax.scatter(xs, ys, c=colors[i], alpha=0.6, zorder=2)

    # Plot means
    means = [scores[col].mean() for col in cols]
    ax.scatter(range(len(cols)), means, c='k', marker='x', s=80, zorder=3)

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('Subject PCA Scores')
    ax.grid(alpha=0.3, zorder=-10)


def plot_pca_reconstruction_ve(reconstruction_ve, ax=None):
    """Plot PCA reconstruction variance explained for each subject.

    Parameters:
        reconstruction_ve (pd.DataFrame) - DataFrame with ve_pc1, ve_pc1_pc2, ve_pc1_pc2_pc3 columns.
        ax (matplotlib.axes.Axes) - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    labels = ['pc1', 'pc1-pc2', 'pc1-pc2-pc3']
    cols = ['ve_pc1', 've_pc1_pc2', 've_pc1_pc2_pc3']

    ve_1 = reconstruction_ve[cols[0]].values
    ve_2 = reconstruction_ve[cols[1]].values
    ve_3 = reconstruction_ve[cols[2]].values

    # Sort on full reconstruction VE
    inds = np.argsort(ve_3)

    ax.plot(ve_1[inds], '.', alpha=0.7, label=labels[0])
    ax.plot(ve_2[inds], '.', alpha=0.7, label=labels[1])
    ax.plot(ve_3[inds], '.', alpha=0.7, label=labels[2])
    ax.set_xlabel('Sorted Subject Number')
    ax.set_ylabel('Variance Explained')
    ax.set_title('PCA Reconstruction VE')
    ax.legend()


def plot_bar_with_arrows(base_vec, arrow_vecs, labels=None, title=None, ax=None):
    """Plot bars with arrow overlays showing component contributions."""
    if ax is None: fig, ax = plt.subplots()


    if labels is None:
        labels = [None for _ in arrow_vecs]

    xs = np.arange(-1, 4)
    xoffsets = 2 * (np.linspace(0, 1, len(arrow_vecs)) - 0.5) * 0.2

    ax.bar(xs, base_vec, alpha=0.2, color='k', label=labels[0])

    s = 0.5
    upper_y_vals = np.array([base_vec + v * s for v in arrow_vecs])
    corder = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Loop over arrow components
    for i in range(len(arrow_vecs)):

        # For each x-value, draw from bar top to y-value
        for j in range(len(xs)):

            # Get line segments for the jth x-value and ith component
            xvals = [xs[j], xs[j]] + xoffsets[i]
            yvals = [base_vec[j], upper_y_vals[i, j]]

            # Draw line segments
            cur_label = labels[i + 1] if j == 0 else None
            ax.plot(xvals, yvals, '-', color=corder[i], label=cur_label)

            # Decide if endpoint marker should point up or down
            marker = '^' if yvals[1] > base_vec[j] else 'v'

            # Draw endpoint markers
            ax.scatter(xvals[1], yvals[1], marker=marker, color=corder[i])

    ax.legend()
    ax.set_xlabel('Peri-CP Trial')
    ax.set_ylabel('Learning Rate')
    if title is not None:
        ax.set_title(title)


def _plot_reliability(reliabilities_by_dataset, dataset_config, col_filter, label_fn, title, ax=None):
    """Plot split-half reliability for multiple datasets.

    Parameters:
        reliabilities_by_dataset (dict) - dataset_name -> DataFrame
        dataset_config (dict)           - dataset_name -> (label, color)
        col_filter (callable)           - Function to filter columns from DataFrame
        label_fn (callable)             - Function to generate x-axis labels from columns
        title (str)                     - Plot title
        ax (matplotlib.axes.Axes)       - Axes to plot on. If None, creates new figure.
    """
    if ax is None: fig, ax = plt.subplots()

    datasets = list(reliabilities_by_dataset.keys())
    n_datasets = len(datasets)

    # Get columns from first dataset
    first_df = reliabilities_by_dataset[datasets[0]]
    cols = [c for c in first_df.columns if col_filter(c)]
    labels = label_fn(cols)
    n_cols = len(cols)

    width = 0.8 / n_datasets
    show_legend = n_datasets > 1

    for d_idx, dataset in enumerate(datasets):
        df = reliabilities_by_dataset[dataset]
        label, color = dataset_config[dataset]

        means = df[cols].mean().values
        sds = df[cols].std().values

        xs = np.arange(n_cols) + (d_idx - (n_datasets - 1) / 2) * width
        ax.errorbar(xs, means, yerr=2*sds, fmt='o', capsize=4, color=color, label=label if show_legend else None)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Split-Half Correlation')
    ax.set_ylim(-0.2, 1.2)
    ax.set_title(title)
    if show_legend: ax.legend(fontsize=8)
    ax.grid(alpha=0.2)


def plot_score_reliability(reliabilities, ax=None):
    """Plot split-half reliability of PCA scores as mean +/- 2 SD."""
    _plot_reliability(
        {'_': reliabilities}, {'_': ('', 'C0')},
        col_filter=lambda c: c.startswith('Rho Score'),
        label_fn=lambda cols: [f'PC{i+1}' for i in range(len(cols))],
        title='PCA Score Reliability', ax=ax
    )


def plot_beta_reliability(reliabilities, ax=None):
    """Plot split-half reliability of regression betas as mean +/- 2 SD."""
    _plot_reliability(
        {'_': reliabilities}, {'_': ('', 'C0')},
        col_filter=lambda c: 'beta_' in c,
        label_fn=lambda cols: [c.split('beta_')[-1].upper() for c in cols],
        title='Regression Beta Reliability', ax=ax
    )


def plot_score_reliability_multi(reliabilities_by_dataset, dataset_config, ax=None):
    """Plot split-half reliability of PCA scores for multiple datasets."""
    _plot_reliability(
        reliabilities_by_dataset, dataset_config,
        col_filter=lambda c: c.startswith('Rho Score'),
        label_fn=lambda cols: [f'PC{i+1}' for i in range(len(cols))],
        title='PCA Score Reliability', ax=ax
    )


def plot_beta_reliability_multi(reliabilities_by_dataset, dataset_config, ax=None):
    """Plot split-half reliability of regression betas for multiple datasets."""
    _plot_reliability(
        reliabilities_by_dataset, dataset_config,
        col_filter=lambda c: 'beta_' in c,
        label_fn=lambda cols: [c.split('beta_')[-1].upper() for c in cols],
        title='Regression Beta Reliability', ax=ax
    )


def cdf(x):
    inds = np.argsort(x)
    frac = np.arange(1,len(x)+1)/ (len(x)+1)
    vals = x[inds]

    return vals, frac

