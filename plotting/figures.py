"""Functions to generate main manuscript figures."""

from configs import *
from changepoint.subjects import *
from analysis.analysis import *
from plotting.plots_basic import *
from plotting.plots_recovery import *
from plotting.plots_comparison import *
from plotting.plots_alt import *
from plotting.svgtools import *

import os

def figure_1(subjs, tasks, subj_pcp_lr, subj_pca_scores, savefig=True, close=True):
    """Generate Figure 1: Task explanation and example adaptive vs non-adaptive subjects.

    Figure 1 content:
        1A. Task diagram (external SVG)
        1B. Non-adaptive subject: task performance
        1C. Non-adaptive subject: PE vs update scatter
        1D. Non-adaptive subject: peri-CP learning rates

        1E. Task logic diagram (external SVG)
        1F. Adaptive subject: task performance
        1G. Adaptive subject: PE vs update scatter
        1H. Adaptive subject: peri-CP learning rates
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    # Use absolute PC0 scores to find most and least adaptive subjects
    # Matches original selection: high abs(PC0) = low adaptive, low abs(PC0) = high adaptive
    scores_abs = np.abs(subj_pca_scores['Score_0'].values)

    # Non-adaptive: highest absolute Score_0
    snum_nonadapt = np.argmax(scores_abs)

    # Adaptive: lowest absolute Score_0
    snum_adapt = np.argmin(scores_abs)

    print(f"Non-adaptive subject: {snum_nonadapt} (|Score_0| = {scores_abs[snum_nonadapt]:.3f})")
    print(f"Adaptive subject: {snum_adapt} (|Score_0| = {scores_abs[snum_adapt]:.3f})")

    # Panel C: Non-adaptive subject task performance
    fig, ax = plt.subplots()
    plot_task_example(subjs, tasks, snum=snum_nonadapt, ax=ax)
    ax.set_title('Example Non-Adaptive Subject Behavior')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig1_C' + FIG_FMT, dpi=300)

    # Panel D: Non-adaptive subject PE vs update
    fig, ax = plt.subplots()
    plot_subject_update_by_pe(subjs[snum_nonadapt], ax=ax)
    ax.set_title('Updates')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig1_D' + FIG_FMT, dpi=300)

    # Panel E: Non-adaptive subject peri-CP learning rates
    fig, ax = plt.subplots()
    plot_peri_cp_lr_bars(subj_pcp_lr, snum=snum_nonadapt, ax=ax)
    ax.set_title('Learning Rates')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig1_E' + FIG_FMT, dpi=300)

    # Panel F: Adaptive subject task performance
    fig, ax = plt.subplots()
    plot_task_example(subjs, tasks, snum=snum_adapt, ax=ax)
    ax.set_title('Example Adaptive Subject Behavior')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig1_F' + FIG_FMT, dpi=300)

    # Panel G: Adaptive subject PE vs update
    fig, ax = plt.subplots()
    plot_subject_update_by_pe(subjs[snum_adapt], ax=ax)
    ax.set_title('Updates')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig1_G' + FIG_FMT, dpi=300)

    # Panel H: Adaptive subject peri-CP learning rates
    fig, ax = plt.subplots()
    plot_peri_cp_lr_bars(subj_pcp_lr, snum=snum_adapt, ax=ax)
    ax.set_title('Learning Rates')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig1_H' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_2(tasks, subj_pcp_lr, group_pca_basis, subj_pca_lr_scores, subj_linear_models, savefig=True, close=True):
    """Generate Figure 2: Normative model explanation and group data.

    Figure 2 layout:
        Column 1: [A] Normative predictions, [D] CPP/RU/LR traces
        Column 2: [B] LR components (bar+arrows), [E] LR PCs (bar+arrows)
        Column 3: [C] Correlation matrix

    Parameters:
        tasks (list) - List of Task objects.
        subj_pcp_lr (pd.DataFrame) - Peri-CP learning rate slopes for each subject.
        group_pca_basis (pd.DataFrame) - PCA basis vectors (PCs as rows).
        subj_pca_lr_scores (pd.DataFrame) - PCA scores for each subject.
        subj_linear_models (pd.DataFrame) - Linear model results with betas.
        savefig (bool) - Whether to save individual panel figures.
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    # Simulate normative model on first task
    sim = Subject()
    params = DEFAULT_PARAMS_SUBJ.copy()
    params['noise_sd'] = tasks[0].noise_sd[0]
    params['hazard'] = tasks[0].hazard[0]
    simulate_subject(sim, tasks[0].obs, params)

    ntrials = 85

    # --- Panel A: Normative predictions ---
    fig, ax = plt.subplots()
    ax.plot(sim.responses.pred[:ntrials], '-', label='Pred')
    ax.plot(tasks[0].obs[:ntrials], '.', label='Obs')
    ax.legend(loc='lower right')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Obs./Pred.')
    ax.set_title('Normative Predictions')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig2_A' + FIG_FMT, dpi=300)

    # --- Panel D: CPP, RU, LR traces ---
    fig, ax = plt.subplots()
    ax.plot(sim.beliefs.cpp[:ntrials], label='CPP')
    ax.plot(sim.beliefs.relunc[:ntrials], label='RU')
    ax.plot(sim.responses.lr[:ntrials], 'k--', label='LR', alpha=0.5)
    ax.legend(loc='upper right')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('CPP/RU')
    ax.set_title('Normative CPP, RU, and LR')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig2_D' + FIG_FMT, dpi=300)

    # --- Panel B: Learning rate components from normative model ---
    # Get peri-CP stats for the normative simulation
    sim_pcp_lr, sim_pcp_cpp, sim_pcp_ru = get_peri_cp_stats([sim], [tasks[0]])

    # Base learning rate (constant component) and modulations from CPP, RU
    base_lr = np.ones(len(sim_pcp_lr.columns)) * 0.1
    cpp_contrib = sim_pcp_cpp.values[0]
    ru_contrib = sim_pcp_ru.values[0]

    fig, ax = plt.subplots()
    plot_bar_with_arrows(
        base_lr,
        [np.ones(len(base_lr)) * 0.05, cpp_contrib, ru_contrib],
        labels=['Base', 'Const', 'CPP', 'RU'],
        title='Learning Rate Components',
        ax=ax
    )
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig2_B' + FIG_FMT, dpi=300)

    # --- Panel C: Correlation matrix between PC scores and betas ---
    # Build correlation matrix
    score_cols = [c for c in subj_pca_lr_scores.columns if c.startswith('Score_')][:2]
    lm_model = 'model-pe-cpp-ru-prod-deltas'
    lm_terms = get_model_terms(lm_model)
    beta_cols = [f'{lm_model}_beta_{t}' for t in lm_terms]
    beta_cols = [c for c in beta_cols if c in subj_linear_models.columns]

    # Combine into single dataframe for correlation
    corr_data = pd.concat([
        subj_pca_lr_scores[score_cols],
        subj_linear_models[beta_cols]
    ], axis=1)

    # Rename for display
    corr_data.columns = ['PC1', 'PC2'] + [t.upper() for t in lm_terms[:len(beta_cols)]]

    # Compute Spearman correlation matrix
    corr_matrix = corr_data.corr(method='spearman')

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Spearman r')
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title('Parameter Estimate Spearman Correlation Matrix')

    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', fontsize=8)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig2_C' + FIG_FMT, dpi=300)

    # --- Panel E: Learning rate PCs ---
    avg_peri_cp_lr = subj_pcp_lr.mean(axis=0).values
    basis_rows = group_pca_basis.values

    fig, ax = plt.subplots()
    plot_bar_with_arrows(avg_peri_cp_lr, basis_rows[:3], labels=['Avg', 'PC1', 'PC2', 'PC3'], title='Learning Rate PCs', ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig2_E' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_3(subjs, tasks, subj_linear_models, subj_pcp_lr, group_pca_basis, subj_pca_scores, cumulative_ve, savefig=True, close=True):
    """Generate Figure 3: Normative model fit and variance explained.

    Figure 3 layout:
        Row 1: [A] PE vs Update comparison, [B] Beta strip plots, [C] Cumulative VE (PCA + LM)
        Row 2: [D] Trialwise regression VE, [E] PCA reconstruction VE

    Parameters:
        subjs (list) - List of Subject objects with beliefs.
        tasks (list) - List of Task objects.
        subj_linear_models (pd.DataFrame) - Linear model results with betas.
        subj_pcp_lr (pd.DataFrame) - Subject peri-CP learning rates.
        group_pca_basis (pd.DataFrame) - PCA basis vectors (PCs as rows).
        subj_pca_scores (pd.DataFrame) - PCA scores for each subject.
        cumulative_ve (array) - Cumulative variance explained from PCA.
        savefig (bool) - Whether to save individual panel figures.
        close (bool) - Whether to close figures after saving.
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Panel A: PE vs Update comparison (PE-only vs CPP-based) ---
    # PE-only model: constant learning rate
    sim_pe = Subject()
    params_pe = DEFAULT_PARAMS_SUBJ.copy()
    params_pe['noise_sd'] = tasks[0].noise_sd[0]
    params_pe['hazard'] = tasks[0].hazard[0]
    params_pe['beta_pe'] = 1.0   # Constant learning rate
    params_pe['beta_cpp'] = 0.0  # No CPP modulation
    params_pe['beta_ru'] = 0.0   # No RU modulation
    simulate_subject(sim_pe, tasks[0].obs, params_pe)

    # CPP-based model: adaptive learning rate
    sim_cpp = Subject()
    params_cpp = DEFAULT_PARAMS_SUBJ.copy()
    params_cpp['noise_sd'] = tasks[0].noise_sd[0]
    params_cpp['hazard'] = tasks[0].hazard[0]
    params_cpp['beta_pe'] = 0.0   # No constant component
    params_cpp['beta_cpp'] = 1.0  # CPP modulation
    params_cpp['beta_ru'] = 0.0   # No RU modulation
    simulate_subject(sim_cpp, tasks[0].obs, params_cpp)

    fig, ax = plt.subplots()
    plot_pe_update_model_comparison(sim_pe, sim_cpp, ax=ax)
    ax.set_title('PE-Only vs CPP-Based')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_A' + FIG_FMT, dpi=300)

    # --- Panel B: Beta strip plots ---
    fig, ax = plt.subplots()
    plot_subj_betas_strip(subj_linear_models, model='model-pe-cpp-ru-prod-deltas', ax=ax)
    ax.set_title('Subject Beta Coefficients')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_B' + FIG_FMT, dpi=300)

    # --- Panel C: Cumulative VE (PCA and linear model) ---
    cumulative_ve_lm = get_lm_cumulative_ve(subjs)
    fig, ax = plt.subplots()
    plot_cumulative_ve(cumulative_ve, cumulative_ve_lm, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_C' + FIG_FMT, dpi=300)

    # --- Panel D: Trialwise regression VE ---
    fig, ax = plt.subplots(figsize=(FIG_WIDE_WIDTH, FIG_STD_HEIGHT))
    plot_lm_ve(subjs, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_D' + FIG_FMT, dpi=300)

    # --- Panel E: PCA reconstruction VE ---
    reconstruction_ve = get_subj_pca_ve(subj_pcp_lr, group_pca_basis, subj_pca_scores)
    fig, ax = plt.subplots(figsize=(FIG_WIDE_WIDTH, FIG_STD_HEIGHT))
    plot_pca_reconstruction_ve(reconstruction_ve, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_E' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_4(reliabilities, savefig=True, close=True):
    """Generate Figure 4: Split-half reliability of parameter estimates.

    Figure 4 layout:
        [A] PCA score reliability
        [B] Regression beta reliability

    Parameters:
        reliabilities (pd.DataFrame) - DataFrame from do_split_half_analysis.
        savefig (bool) - Whether to save individual panel figures.
        close (bool) - Whether to close figures after saving.
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Panel A: PCA score reliability ---
    fig, ax = plt.subplots()
    plot_score_reliability(reliabilities, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig4_A' + FIG_FMT, dpi=300)

    # --- Panel B: Regression beta reliability ---
    fig, ax = plt.subplots()
    plot_beta_reliability(reliabilities, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig4_B' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_5(err_analysis, fim_df, recovery_analysis, savefig=True, close=True):
    """Generate Figure 5: Parameter recovery and estimation analysis.

    Figure 5 layout:
        Row 1: [A] beta_pe recovery, [B] beta_cpp recovery, [C] beta_ru recovery
        Row 2: [D] Error correlation matrix, [E] Task scatter, [F] Variance decomposition

    Parameters:
        err_analysis (dict) - Output from analyze_error_covariance.
        fim_df (pd.DataFrame) - Output from analyze_task_information.
        recovery_analysis (dict) - Output from analyze_recovery.
        savefig (bool) - Whether to save individual panel figures.
        close (bool) - Whether to close figures after saving.
    """
    from plotting.plots_fim import plot_task_scatter
    from plotting.plots_recovery import plot_error_corr_matrix, plot_variance_decomposition, plot_param_recovery

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Row 1: Parameter recovery plots ---

    # Panel A: beta_pe recovery
    fig, ax = plt.subplots()
    plot_param_recovery(recovery_analysis, 'beta_pe', ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig5_A' + FIG_FMT, dpi=300)

    # Panel B: beta_cpp recovery
    fig, ax = plt.subplots()
    plot_param_recovery(recovery_analysis, 'beta_cpp', ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig5_B' + FIG_FMT, dpi=300)

    # Panel C: beta_ru recovery
    fig, ax = plt.subplots()
    plot_param_recovery(recovery_analysis, 'beta_ru', ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig5_C' + FIG_FMT, dpi=300)

    # --- Row 2: Estimation analysis ---

    # Panel D: Error correlation matrix
    fig, ax = plt.subplots()
    im = plot_error_corr_matrix(err_analysis, ax=ax)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig5_D' + FIG_FMT, dpi=300)

    # Panel E: N changepoints vs std run length, colored by max error SD
    fig, ax = plt.subplots()
    sc = plot_task_scatter(fim_df, 'n_changepoints', 'std_run_length', color_col='err_sd_2', ax=ax)
    fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig5_E' + FIG_FMT, dpi=300)

    # Panel F: Error source decomposition
    fig, ax = plt.subplots()
    plot_variance_decomposition(recovery_analysis, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig5_F' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_6(recovery_results, savefig=True, close=True):
    """Generate Figure 6: Recovery SD as a function of beta_pe.

    Figure 6 layout:
        Row 1: [A] SD(beta_pe) vs beta_pe, [B] SD(beta_cpp) vs beta_pe, [C] SD(beta_ru) vs beta_pe
        Row 2: [D] Heatmap beta_pe,         [E] Heatmap beta_cpp,         [F] Heatmap beta_ru

    Parameters:
        recovery_results (dict) - Output from parameter_recovery (with 'results', 'true_params').
        savefig (bool)          - Whether to save individual panel figures.
        close (bool)            - Whether to close figures after saving.
    """
    from plotting.plots_recovery import plot_recovery_sd_by_param, plot_recovery_sd_heatmap

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    results = recovery_results['results']
    true_params = recovery_results['true_params']
    param_names = results['param_names']
    params = ['beta_pe', 'beta_cpp', 'beta_ru']
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Row 1: line plots
    for i, p in enumerate(params):
        fig, ax = plt.subplots()
        plot_recovery_sd_by_param(results, true_params, param_names, y_param=p, ax=ax)
        fig.tight_layout()
        if savefig: fig.savefig(FIGURES_DIR + f'fig6_{panel_labels[i]}' + FIG_FMT, dpi=300)

    # Row 2: heatmaps
    for i, p in enumerate(params):
        fig, ax = plt.subplots()
        im = plot_recovery_sd_heatmap(results, true_params, param_names, y_param=p, ax=ax)
        fig.colorbar(im, ax=ax, label='SD')
        fig.tight_layout()
        if savefig: fig.savefig(FIGURES_DIR + f'fig6_{panel_labels[i + 3]}' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_7(recovery_results, savefig=True, close=True):
    """Generate Figure 7: Recovery SD as a function of beta_pe (high beta_pe range).

    Same layout as Figure 6, but for beta_pe in [0.8, 0.98].

    Figure 7 layout:
        Row 1: [A] SD(beta_pe) vs beta_pe, [B] SD(beta_cpp) vs beta_pe, [C] SD(beta_ru) vs beta_pe
        Row 2: [D] Heatmap beta_pe,         [E] Heatmap beta_cpp,         [F] Heatmap beta_ru

    Parameters:
        recovery_results (dict) - Output from parameter_recovery (with 'results', 'true_params').
        savefig (bool)          - Whether to save individual panel figures.
        close (bool)            - Whether to close figures after saving.
    """
    from plotting.plots_recovery import plot_recovery_sd_by_param, plot_recovery_sd_heatmap

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    results = recovery_results['results']
    true_params = recovery_results['true_params']
    param_names = results['param_names']
    params = ['beta_pe', 'beta_cpp', 'beta_ru']
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Row 1: line plots
    for i, p in enumerate(params):
        fig, ax = plt.subplots()
        plot_recovery_sd_by_param(results, true_params, param_names, y_param=p, ax=ax)
        fig.tight_layout()
        if savefig: fig.savefig(FIGURES_DIR + f'fig7_{panel_labels[i]}' + FIG_FMT, dpi=300)

    # Row 2: heatmaps
    for i, p in enumerate(params):
        fig, ax = plt.subplots()
        im = plot_recovery_sd_heatmap(results, true_params, param_names, y_param=p, ax=ax)
        fig.colorbar(im, ax=ax, label='SD')
        fig.tight_layout()
        if savefig: fig.savefig(FIGURES_DIR + f'fig7_{panel_labels[i + 3]}' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_8(comparison, savefig=True, close=True):
    """Generate Figure 8: Model comparison (recovery, reliability, variance explained).

    Figure 8 layout:
        [A] Recovery r by parameter and model
        [B] Split-half reliability by parameter and model
        [C] Variance explained by model

    Parameters:
        comparison (dict) - Output from model_comparison_analysis.
        savefig (bool)    - Whether to save individual panel figures.
        close (bool)      - Whether to close figures after saving.
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    # Panel A: Recovery correlation by parameter and model
    fig, ax = plt.subplots()
    plot_recovery_by_model(comparison['models'], comparison['recovery'], comparison['gen_params'], ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig8_A' + FIG_FMT, dpi=300)

    # Panel B: Split-half reliability by parameter and model
    fig, ax = plt.subplots()
    plot_reliability_by_model(comparison['models'], comparison['reliability'], ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig8_B' + FIG_FMT, dpi=300)

    # Panel C: Variance explained by model
    fig, ax = plt.subplots()
    plot_ve_by_model(comparison['models'], comparison['ve'], ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig8_C' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def figure_9_alt(alt_analysis, real_lm, savefig=True, close=True):
    """Generate Figure 9: Alternative cognitive models comparison.

    Figure 9 layout:
        [A] Beta comparison (alternative models vs real data)
        [B] Variance explained comparison
        [C] Example subject traces (2 rows x 3 cols)

    Parameters:
        alt_analysis (dict) - Output from alt_model_analysis with 'results', 'tasks', 'lm_model'.
        real_lm (pd.DataFrame) - Linear model results for real data.
        savefig (bool) - Whether to save individual panel figures.
        close (bool) - Whether to close figures after saving.
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    results = alt_analysis['results']
    tasks = alt_analysis['tasks']
    lm_model = alt_analysis['lm_model']

    # Panel A: Beta comparison
    fig, ax = plt.subplots()
    plot_comparison_betas(results, real_lm, lm_model, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig9_A' + FIG_FMT, dpi=300)

    # Panel B: VE comparison
    fig, ax = plt.subplots()
    plot_comparison_ve(results, real_lm, lm_model, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig9_B' + FIG_FMT, dpi=300)

    # Panel C: Example subjects (2 rows x 3 cols)
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(16, 3.5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    plot_example_subjects(results, tasks, lm_model, ax_grid=axes)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig9_C' + FIG_FMT, dpi=300)

    if close: plt.close('all')
