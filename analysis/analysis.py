"""Functions for fitting models to subject data."""
from configs import *
from analysis.aggregation import *

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import linregress


def get_peri_cp_stats(subjs, tasks, endpoint=4):
    """Compute peri-changepoint statistics for all subject/task pairs.

    Returns:
        tuple of (subj_pcp_lr, subj_pcp_cpp, subj_pcp_ru) DataFrames with columns
        labeled 'CP-1', 'CP0', 'CP+1', etc.
    """

    # Get peri-cp trial positions from first task
    _, pcp_trials = align_trials(tasks[0], endpoint)
    n_pcp = len(pcp_trials)
    n_subj = len(subjs)

    # Build column names from peri-cp trial positions
    col_names = [f'CP{t:+d}' if t != 0 else 'CP0' for t in pcp_trials]

    # Initialize output arrays
    lr_data  = np.full((n_subj, n_pcp), np.nan)
    cpp_data = np.full((n_subj, n_pcp), np.nan)
    ru_data  = np.full((n_subj, n_pcp), np.nan)

    # Compute stats for each subject
    for i, (subj, task) in enumerate(zip(subjs, tasks)):

        # Get pre/post indices for filtering
        pre, pst = get_relinds(task.cp, endpoint=endpoint)

        # Compute statistics for each peri-cp position
        for j, pcp_pos in enumerate(pcp_trials):

            # Filter trial indices for this peri-cp position
            if pcp_pos < 0:
                # Pre-CP trials should not be immediately after some other CP
                inds = np.where((pre == pcp_pos) & ~((pst == 1) | (pst == 2)))[0]
            else:
                inds = np.where(pst == pcp_pos)[0]

            # Regression of update on prediction error
            pe = subj.responses.pe[inds]
            up = subj.responses.update[inds]
            slope, _, _, _, _ = linregress(pe, up)
            lr_data[i, j] = slope

            # Belief averages if available
            if subj.beliefs is not None:
                ru_data[i, j]  = np.nanmean(subj.beliefs.relunc[inds])
                cpp_data[i, j] = np.nanmean(subj.beliefs.cpp[inds])

    # Convert to DataFrames
    subj_pcp_lr  = pd.DataFrame(lr_data , columns=col_names)
    subj_pcp_cpp = pd.DataFrame(cpp_data, columns=col_names)
    subj_pcp_ru  = pd.DataFrame(ru_data , columns=col_names)

    return subj_pcp_lr, subj_pcp_cpp, subj_pcp_ru


def fit_linear_models(subjs, model='full'):
    """Fit linear models to each subject's data and return DataFrame with betas, VIFs, and variance explained.

    Returns:
        pd.DataFrame with subjects as rows and columns for each beta, vif, and model ve.
    """

    # Define term names for each model type
    term_names = {
        'm0': ['c', 'pe'],
        'm1': ['c', 'pe', 'cpp'],
        'm2': ['c', 'pe', 'cpp', 'ru'],
        'm3': ['c', 'pe', 'cpp', 'ru', 'prod'],
        'm4': ['c', 'pe', 'cppd', 'rud', 'prodd'],
        'n0': ['c', 'cpp', 'prod'],
        'n1': ['c', 'cpp', 'ru', 'prod'],
    }

    terms = term_names[model]
    rows = []

    for subj in subjs:

        # Aliases to simplify code
        pe  = subj.responses.pe
        up  = subj.responses.update
        cpp = subj.beliefs.cpp
        ru = subj.beliefs.relunc

        # Define deviation variables
        dcpp = cpp - np.nanmean(cpp)
        dru = ru - np.nanmean(ru)

        # Define intercept for design matrices
        const = np.ones(len(pe))

        # Design matrices

        match model:
            case 'm0':
                design = np.vstack([const, pe]).T
            case 'm1':
                design = np.vstack([const, pe, pe*cpp]).T
            case 'm2':
                design = np.vstack([const, pe, pe*cpp, pe*ru]).T
            case 'm3':
                design = np.vstack([const, pe, pe*cpp, pe*ru, pe*cpp*ru]).T
            case 'm4':
                design = np.vstack([const, pe, pe*dcpp, pe*dru, pe*dcpp*dru]).T
            case 'n0':
                design = np.vstack([const, pe*cpp, pe*ru*(1-cpp)]).T
            case 'n1':
                design = np.vstack([const, pe*cpp, pe*ru, pe*cpp*ru]).T

        # Fit model
        results = sm.OLS(up, design).fit()

        # Build row dict with model prefix
        row = {}
        for i, term in enumerate(terms):
            row[f'{model}_beta_{term}'] = results.params[i]
            row[f'{model}_vif_{term}'] = variance_inflation_factor(design, i)
        row[f'{model}_ve'] = results.rsquared

        rows.append(row)

    return pd.DataFrame(rows)


def fit_pca(df, standardize_signs=True):
    """Perform PCA on data matrix, returning eigenvalues, eigenvectors, and scores."""

    # Compute the eigendecomposition of the covariance matrix
    vals = np.array(df)
    evals, evecs = np.linalg.eig(np.cov(vals.T))

    # Reorder eigenvalues and eigenvectors in descending order
    inds = np.flip(np.argsort(evals))
    evals = evals[inds]
    evecs = evecs[:,inds]

    # Standardize eigenvec signs
    if standardize_signs:
        # Ensure the first eigenvector is positive in the first element
        evecs[:,0] = -evecs[:,0] if evecs[0,0] < 0 else evecs[:,0]

        # Ensure the second eigenvector is positive in the second element
        evecs[:,1] = -evecs[:,1] if evecs[1,1] < 0 else evecs[:,1]

        # Ensure the third eigenvector is positive in the third element
        evecs[:,2] = -evecs[:,2] if evecs[2,2] < 0 else evecs[:,2]

    scores = (vals - np.mean(vals, axis=0)) @ evecs
    return evals, evecs, scores


def fit_peri_cp_pca(subj_pcp_lr):
    """Perform PCA on peri-changepoint learning rate slopes.

    Parameters:
        subj_pcp_lr (pd.DataFrame) - DataFrame of peri-changepoint learning rate slopes.

    Returns:
        tuple of (basis, scores, cumulative_ve):
            - basis: DataFrame with principal components as rows, peri-CP positions as columns
            - scores: DataFrame with subjects as rows, PC scores as columns
            - cumulative_ve: array of cumulative fractional variance explained by each PC
    """

    # Perform PCA on the peri-changepoint slopes
    evals, evecs, scores = fit_pca(subj_pcp_lr, standardize_signs=True)

    # Convert to standard row-major format
    basis_rows = evecs.T

    # Cumulative fractional variance explained
    cumulative_ve = np.cumsum(evals) / np.sum(evals)

    # Convert to dataframes
    scores = pd.DataFrame(scores, columns=['Score_' + str(i) for i in range(len(evals))])
    basis  = pd.DataFrame(basis_rows, columns=subj_pcp_lr.columns)

    return basis, scores, cumulative_ve


def get_subj_pca_ve(subj_pcp_lr, group_pca_basis, subj_pca_scores):
    """Compute variance explained by PCA reconstruction at each level.

    For each subject, reconstructs their peri-CP LR curve using 1, 2, or 3 PCs
    and computes R-squared (1 - SS_residual / SS_total) relative to the group mean.

    Parameters:
        subj_pcp_lr (pd.DataFrame) - Subject peri-CP learning rates (subjects x positions).
        group_pca_basis (pd.DataFrame) - PCA basis vectors (PCs as rows).
        subj_pca_scores (pd.DataFrame) - PCA scores for each subject.

    Returns:
        pd.DataFrame with columns 've_pc1', 've_pc1_pc2', 've_pc1_pc2_pc3'.
    """
    n_subj = len(subj_pcp_lr)
    actual = subj_pcp_lr.values
    mean_lr = actual.mean(axis=0)
    basis = group_pca_basis.values

    # Get score columns
    score_cols = [c for c in subj_pca_scores.columns if c.startswith('Score_')]
    scores = subj_pca_scores[score_cols].values

    # Compute VE for each reconstruction level
    ve_data = np.zeros((n_subj, 3))

    for i in range(n_subj):
        actual_i = actual[i]

        # Total sum of squares relative to group mean
        ss_total = np.sum((actual_i - mean_lr) ** 2)

        # Reconstruct with 1, 2, 3 PCs
        for n_pcs in range(1, 4):
            recon = mean_lr.copy()
            for pc in range(n_pcs):
                recon = recon + scores[i, pc] * basis[pc]

            # Residual sum of squares
            ss_residual = np.sum((actual_i - recon) ** 2)
            ve_data[i, n_pcs - 1] = 1 - ss_residual / ss_total if ss_total > 0 else 0

    return pd.DataFrame(ve_data, columns=['ve_pc1', 've_pc1_pc2', 've_pc1_pc2_pc3'])


def get_lm_cumulative_ve(subjs):
    """Compute cumulative variance explained by nested linear models.

    Fits models m0 (pe), m1 (pe+cpp), m2 (pe+cpp+ru) and returns mean VE across subjects.

    Returns:
        array of length 3 with mean VE for each model level.
    """
    models = ['m0', 'm1', 'm2']
    cumulative_ve = np.zeros(len(models))

    for i, model in enumerate(models):
        lm_results = fit_linear_models(subjs, model=model)
        cumulative_ve[i] = lm_results[f'{model}_ve'].mean()

    return cumulative_ve
