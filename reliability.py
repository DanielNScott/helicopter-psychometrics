"""Within and between subject reliability analyses."""
from configs import *
from analysis import fit_linear_models, fit_peri_cp_pca, get_peri_cp_stats
from aggregation import split_within_subjects


def _get_split_half_reliabilities(subjs, tasks, analysis_fn, nreps=20):
    """Perform within-subject split-half reliability analysis."""
    # Initialize results dict
    results = []

    # Repeat split-half analysis over random splits
    for rep in range(nreps):
        print(f'Running within-subject split-half analysis {rep+1}/{nreps}')
       
        # Split the data
        subjs_a, subjs_b, tasks_a, tasks_b = split_within_subjects(subjs, tasks)

        # Run the analysis function
        results.append(analysis_fn(subjs_a, subjs_b, tasks_a, tasks_b))

    # Return data frame with columns as results, rows as repetitions
    return pd.DataFrame(results)


def _analysis_pca(subjs_a, subjs_b, tasks_a, tasks_b):
    """Compute PCA basis cosines and score correlations between two splits."""

    # Get peri-CP learning rate slopes for each split
    pcp_lr_a, _, _ = get_peri_cp_stats(subjs_a, tasks_a)
    pcp_lr_b, _, _ = get_peri_cp_stats(subjs_b, tasks_b)

    # Fit PCA to each split
    basis_a, scores_a, _ = fit_peri_cp_pca(pcp_lr_a)
    basis_b, scores_b, _ = fit_peri_cp_pca(pcp_lr_b)

    # # Cosines between basis vectors
    # pca_cosines = np.array([
    #     inner_angle(basis_a.values[0], basis_b.values[0]),
    #     inner_angle(basis_a.values[1], basis_b.values[1]),
    #     inner_angle(basis_a.values[2], basis_b.values[2])
    # ])

    # Correlations between scores
    pca_corrs = np.array([
        np.corrcoef(scores_a['Score_0'].values, scores_b['Score_0'].values)[0,1],
        np.corrcoef(scores_a['Score_1'].values, scores_b['Score_1'].values)[0,1],
        np.corrcoef(scores_a['Score_2'].values, scores_b['Score_2'].values)[0,1]
    ])

    return {'Rho Score 0': pca_corrs[0], 'Rho Score 1': pca_corrs[1], 'Rho Score 2': pca_corrs[2]}


def _analysis_regression(subjs_a, subjs_b, tasks_a, tasks_b, model='n0'):
    """Compute regression coefficient correlations between two splits."""

    # Fit models
    lm_a = fit_linear_models(subjs_a, model=model)
    lm_b = fit_linear_models(subjs_b, model=model)

    # Get beta columns (exclude vif and ve columns)
    beta_cols = [c for c in lm_a.columns if f'{model}_beta_' in c]

    # Correlations between betas
    reg_corrs = np.array([np.corrcoef(lm_a[col].values, lm_b[col].values)[0,1] for col in beta_cols])

    # Return as dict
    return {f'Rho {col}': reg_corrs[k] for k, col in enumerate(beta_cols)}


def do_split_half_analysis(subjs, tasks, nreps=20):
    """Wrapper for performing all of the split half reliabilities on same splits."""

    # Package analysis functions as one
    def analysis_fn(subjs_a, subjs_b, tasks_a, tasks_b):
        results = {}
        results.update(       _analysis_pca(subjs_a, subjs_b, tasks_a, tasks_b))
        results.update(_analysis_regression(subjs_a, subjs_b, tasks_a, tasks_b, model='n0'))
        return results
    
    # Run split-half reliability analysis
    return _get_split_half_reliabilities(subjs, tasks, analysis_fn, nreps=nreps)


def inner_angle(a,b):
    # Normalize vectors
    a_hat = a/np.linalg.norm(a)
    b_hat = b/np.linalg.norm(b)

    # Compute dot product
    return np.dot(a_hat,b_hat)
