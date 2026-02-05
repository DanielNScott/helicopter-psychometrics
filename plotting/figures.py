"""Functions to generate main manuscript figures."""

from configs import *
from changepoint.subjects import *
from analysis.analysis import *
from plotting.plots_basic import *
from plotting.plots_recovery import *
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


def compile_figure_1(cleanup=FIG_CLEANUP):
    """Compile Figure 1 from individual panels.

    Layout:
        Row 1: [C] [D] [E]  (non-adaptive subject panels)
        Row 2: [F] [G] [H]  (adaptive subject panels)

    When A and E (manual SVGs) are ready, layout will become:
        Row 1: [A] [C] [D] [E]
        Row 2: [B] [F] [G] [H]
    """
    # Input panel files
    panel_c = FIGURES_DIR + 'fig1_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig1_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig1_E' + FIG_FMT
    panel_f = FIGURES_DIR + 'fig1_F' + FIG_FMT
    panel_g = FIGURES_DIR + 'fig1_G' + FIG_FMT
    panel_h = FIGURES_DIR + 'fig1_H' + FIG_FMT

    # Intermediate files
    row1_cd = FIGURES_DIR + 'fig1_row1_cd.svg'
    row1 = FIGURES_DIR + 'fig1_row1.svg'
    row2_fg = FIGURES_DIR + 'fig1_row2_fg.svg'
    row2 = FIGURES_DIR + 'fig1_row2.svg'
    combined = FIGURES_DIR + 'fig1_combined.svg'
    labeled = FIGURES_DIR + 'fig1_labeled.svg'
    final = FIGURES_DIR + 'fig1_final.svg'

    # Merge row 1: C + D + E
    combine_svgs_horizontal(panel_c, panel_d, row1_cd)
    combine_svgs_horizontal(row1_cd, panel_e, row1)

    # Merge row 2: F + G + H
    combine_svgs_horizontal(panel_f, panel_g, row2_fg)
    combine_svgs_horizontal(row2_fg, panel_h, row2)

    # Merge rows vertically
    combine_svgs_vertical(row1, row2, combined)

    # Add panel labels
    # Standard panel: 460.8 x 345.6 pt (6.4 x 4.8 inches at 72 dpi)
    # Row 1: C at x=0, D at x=460.8, E at x=921.6
    # Row 2: F at x=0, G at x=460.8, H at x=921.6, y offset by 345.6
    pw = 460.8  # panel width
    ph = 345.6  # panel height
    add_text_to_svg(combined, labeled, 'C', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=2*pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'F', x=10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'G', x=pw+10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'H', x=2*pw+10, y=ph+20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_c, panel_d, panel_e, panel_f, panel_g, panel_h]
        intermediate_files = [row1_cd, row1, row2_fg, row2, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


def compile_figure_2(cleanup=FIG_CLEANUP):
    """Compile Figure 2 from individual panels.

    Layout:
        Column 1: [A] above [D]  (normative predictions, CPP/RU/LR)
        Column 2: [B] above [E]  (LR components, LR PCs)
        Column 3: [C]            (correlation matrix)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig2_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig2_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig2_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig2_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig2_E' + FIG_FMT

    # Intermediate files
    col1 = FIGURES_DIR + 'fig2_col1.svg'
    col2 = FIGURES_DIR + 'fig2_col2.svg'
    cols_12 = FIGURES_DIR + 'fig2_cols12.svg'
    combined = FIGURES_DIR + 'fig2_combined.svg'
    labeled = FIGURES_DIR + 'fig2_labeled.svg'
    final = FIGURES_DIR + 'fig2_final.svg'

    # Build column 1: A above D
    combine_svgs_vertical(panel_a, panel_d, col1)

    # Build column 2: B above E
    combine_svgs_vertical(panel_b, panel_e, col2)

    # Merge columns horizontally: col1 + col2 + C
    combine_svgs_horizontal(col1, col2, cols_12)
    combine_svgs_horizontal(cols_12, panel_c, combined)

    # Add panel labels
    # Layout: col1 (A/D) + col2 (B/E) + C
    # Standard panel: 460.8 x 345.6 pt
    pw = 460.8  # panel width
    ph = 345.6  # panel height
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=pw+10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=2*pw+10, y=20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e]
        intermediate_files = [col1, col2, cols_12, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


def compile_figure_3(cleanup=FIG_CLEANUP):
    """Compile Figure 3 from individual panels.

    Layout:
        Row 1: [A] [B] [C]  (PE comparison, beta strips, cumulative VE)
        Row 2: [D] [E]      (regression VE, PCA reconstruction VE)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig3_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig3_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig3_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig3_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig3_E' + FIG_FMT

    # Intermediate files
    row1_ab = FIGURES_DIR + 'fig3_row1_ab.svg'
    row1 = FIGURES_DIR + 'fig3_row1.svg'
    row2 = FIGURES_DIR + 'fig3_row2.svg'
    combined = FIGURES_DIR + 'fig3_combined.svg'
    labeled = FIGURES_DIR + 'fig3_labeled.svg'
    final = FIGURES_DIR + 'fig3_final.svg'

    # Build row 1: A + B + C
    combine_svgs_horizontal(panel_a, panel_b, row1_ab)
    combine_svgs_horizontal(row1_ab, panel_c, row1)

    # Build row 2: D + E
    combine_svgs_horizontal(panel_d, panel_e, row2)

    # Combine rows
    combine_svgs_vertical(row1, row2, combined)

    # Add panel labels
    # Row 1: A, B, C (standard panels)
    # Row 2: D, E (wide panels: 691.2 pt each)
    pw = 460.8   # standard panel width
    ph = 345.6   # panel height
    wpw = 691.2  # wide panel width
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=2*pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=wpw+10, y=ph+20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e]
        intermediate_files = [row1_ab, row1, row2, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


def compile_figure_4(cleanup=FIG_CLEANUP):
    """Compile Figure 4 from individual panels.

    Layout:
        [A] [B]  (score reliability, beta reliability)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig4_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig4_B' + FIG_FMT

    # Intermediate files
    combined = FIGURES_DIR + 'fig4_combined.svg'
    labeled = FIGURES_DIR + 'fig4_labeled.svg'
    final = FIGURES_DIR + 'fig4_final.svg'

    # Merge panels horizontally
    combine_svgs_horizontal(panel_a, panel_b, combined)

    # Add panel labels
    # Two standard panels side by side
    pw = 460.8  # panel width
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b]
        intermediate_files = [combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


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


def compile_figure_5(cleanup=FIG_CLEANUP):
    """Compile Figure 5 from individual panels.

    Layout:
        Row 1: [A] [B] [C]  (beta_pe, beta_cpp, beta_ru recovery)
        Row 2: [D] [E] [F]  (error corr, task scatter, variance decomposition)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig5_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig5_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig5_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig5_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig5_E' + FIG_FMT
    panel_f = FIGURES_DIR + 'fig5_F' + FIG_FMT

    # Intermediate files
    row1_ab = FIGURES_DIR + 'fig5_row1_ab.svg'
    row1 = FIGURES_DIR + 'fig5_row1.svg'
    row2_de = FIGURES_DIR + 'fig5_row2_de.svg'
    row2 = FIGURES_DIR + 'fig5_row2.svg'
    combined = FIGURES_DIR + 'fig5_combined.svg'
    labeled = FIGURES_DIR + 'fig5_labeled.svg'
    final = FIGURES_DIR + 'fig5_final.svg'

    # Build row 1: A + B + C
    combine_svgs_horizontal(panel_a, panel_b, row1_ab)
    combine_svgs_horizontal(row1_ab, panel_c, row1)

    # Build row 2: D + E + F
    combine_svgs_horizontal(panel_d, panel_e, row2_de)
    combine_svgs_horizontal(row2_de, panel_f, row2)

    # Combine rows vertically
    combine_svgs_vertical(row1, row2, combined)

    # Add panel labels
    pw = 460.8  # panel width
    ph = 345.6  # panel height
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=2*pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=pw+10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'F', x=2*pw+10, y=ph+20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]
        intermediate_files = [row1_ab, row1, row2_de, row2, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


def compile_figure_6(cleanup=FIG_CLEANUP):
    """Compile Figure 6 from individual panels.

    Layout:
        Row 1: [A] [B] [C]  (recovery SD line plots)
        Row 2: [D] [E] [F]  (recovery SD heatmaps)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig6_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig6_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig6_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig6_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig6_E' + FIG_FMT
    panel_f = FIGURES_DIR + 'fig6_F' + FIG_FMT

    # Intermediate files
    row1_ab = FIGURES_DIR + 'fig6_row1_ab.svg'
    row1 = FIGURES_DIR + 'fig6_row1.svg'
    row2_de = FIGURES_DIR + 'fig6_row2_de.svg'
    row2 = FIGURES_DIR + 'fig6_row2.svg'
    combined = FIGURES_DIR + 'fig6_combined.svg'
    labeled = FIGURES_DIR + 'fig6_labeled.svg'
    final = FIGURES_DIR + 'fig6_final.svg'

    # Build row 1: A + B + C
    combine_svgs_horizontal(panel_a, panel_b, row1_ab)
    combine_svgs_horizontal(row1_ab, panel_c, row1)

    # Build row 2: D + E + F
    combine_svgs_horizontal(panel_d, panel_e, row2_de)
    combine_svgs_horizontal(row2_de, panel_f, row2)

    # Combine rows vertically
    combine_svgs_vertical(row1, row2, combined)

    # Add panel labels
    pw = 460.8  # panel width
    ph = 345.6  # panel height
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=2*pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=pw+10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'F', x=2*pw+10, y=ph+20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]
        intermediate_files = [row1_ab, row1, row2_de, row2, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


def figure_8(models, recovery_results, reliability_results, ve_results, gen_params, savefig=True, close=True):
    """Generate Figure 8: Model comparison (recovery, reliability, variance explained).

    Figure 8 layout:
        [A] Recovery r by parameter and model
        [B] Split-half reliability by parameter and model
        [C] Variance explained by model

    Parameters:
        models (dict)              - Model name -> param names mapping.
        recovery_results (dict)    - Model name -> recovery result dict.
        reliability_results (dict) - Model name -> reliability DataFrame.
        ve_results (dict)          - Model name -> VE array (per subject).
        gen_params (list)          - Generative parameter names.
        savefig (bool)             - Whether to save individual panel figures.
        close (bool)               - Whether to close figures after saving.
    """

    if savefig: os.makedirs(FIGURES_DIR, exist_ok=True)

    model_labels = {
        'model-pe-cpp-ru':             'pe-cpp-ru',
        'model-pe-cpp-ru-prod':        'pe-cpp-ru-prod',
        'model-pe-cpp-ru-deltas':      'deltas',
        'model-pe-cpp-ru-prod-deltas': 'prod-deltas',
    }
    model_colors = {
        'model-pe-cpp-ru':             'C0',
        'model-pe-cpp-ru-prod':        'C1',
        'model-pe-cpp-ru-deltas':      'C2',
        'model-pe-cpp-ru-prod-deltas': 'C3',
    }

    n_models = len(models)
    width = 0.8 / n_models

    # --- Panel A: Recovery correlation by parameter and model ---
    fig, ax = plt.subplots()
    for m_idx, model in enumerate(models):
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
            ax.plot(x, r, 'o', color=model_colors[model], markersize=7)

    for model in models:
        ax.plot([], [], 'o', color=model_colors[model], label=model_labels[model])
    ax.set_xticks(range(len(gen_params)))
    ax.set_xticklabels(gen_params)
    ax.set_ylabel('Recovery r (true vs recovered)')
    ax.set_title('Parameter Recovery by Model')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig8_A' + FIG_FMT, dpi=300)

    # --- Panel B: Split-half reliability by parameter and model ---

    def _short_beta_name(col, model):
        prefix = f'Rho {model}_'
        return col.replace(prefix, '') if col.startswith(prefix) else col

    all_short_names = []
    for model in models:
        rel = reliability_results[model]
        for col in rel.columns:
            short = _short_beta_name(col, model)
            if short not in all_short_names:
                all_short_names.append(short)

    fig, ax = plt.subplots()
    for m_idx, model in enumerate(models):
        rel = reliability_results[model]
        for col in rel.columns:
            short = _short_beta_name(col, model)
            p_idx = all_short_names.index(short)
            mean_r = rel[col].mean()
            std_r = rel[col].std()
            x = p_idx + (m_idx - (n_models - 1) / 2) * width
            ax.errorbar(x, mean_r, yerr=std_r, fmt='o', color=model_colors[model],
                        markersize=7, capsize=3)

    for model in models:
        ax.plot([], [], 'o', color=model_colors[model], label=model_labels[model])
    ax.set_xticks(range(len(all_short_names)))
    ax.set_xticklabels(all_short_names, rotation=30, ha='right')
    ax.set_ylabel('Split-half r (mean +/- sd)')
    ax.set_title('Parameter Reliability by Model')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig8_B' + FIG_FMT, dpi=300)

    # --- Panel C: Variance explained by model ---
    fig, ax = plt.subplots()
    positions = range(len(models))
    model_names = list(models.keys())

    for m_idx, model in enumerate(model_names):
        ve = ve_results[model]
        mean_ve = ve.mean()
        std_ve = ve.std()
        ax.errorbar(m_idx, mean_ve, yerr=std_ve, fmt='o', color=model_colors[model],
                    markersize=7, capsize=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([model_labels[m] for m in model_names], rotation=30, ha='right')
    ax.set_ylabel('Variance explained (R²)')
    ax.set_title('Model Variance Explained')
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig8_C' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def compile_figure_8(cleanup=FIG_CLEANUP):
    """Compile Figure 8 from individual panels.

    Layout:
        [A] [B] [C]  (recovery, reliability, VE)
    """
    panel_a = FIGURES_DIR + 'fig8_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig8_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig8_C' + FIG_FMT

    # Intermediate files
    row_ab = FIGURES_DIR + 'fig8_row_ab.svg'
    combined = FIGURES_DIR + 'fig8_combined.svg'
    labeled = FIGURES_DIR + 'fig8_labeled.svg'
    final = FIGURES_DIR + 'fig8_final.svg'

    # Build row: A + B + C
    combine_svgs_horizontal(panel_a, panel_b, row_ab)
    combine_svgs_horizontal(row_ab, panel_c, combined)

    # Add panel labels
    pw = 460.8  # panel width
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=2*pw+10, y=20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b, panel_c]
        intermediate_files = [row_ab, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)


def compile_figure_7(cleanup=FIG_CLEANUP):
    """Compile Figure 7 from individual panels.

    Layout:
        Row 1: [A] [B] [C]  (recovery SD line plots, high beta_pe)
        Row 2: [D] [E] [F]  (recovery SD heatmaps, high beta_pe)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig7_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig7_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig7_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig7_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig7_E' + FIG_FMT
    panel_f = FIGURES_DIR + 'fig7_F' + FIG_FMT

    # Intermediate files
    row1_ab = FIGURES_DIR + 'fig7_row1_ab.svg'
    row1 = FIGURES_DIR + 'fig7_row1.svg'
    row2_de = FIGURES_DIR + 'fig7_row2_de.svg'
    row2 = FIGURES_DIR + 'fig7_row2.svg'
    combined = FIGURES_DIR + 'fig7_combined.svg'
    labeled = FIGURES_DIR + 'fig7_labeled.svg'
    final = FIGURES_DIR + 'fig7_final.svg'

    # Build row 1: A + B + C
    combine_svgs_horizontal(panel_a, panel_b, row1_ab)
    combine_svgs_horizontal(row1_ab, panel_c, row1)

    # Build row 2: D + E + F
    combine_svgs_horizontal(panel_d, panel_e, row2_de)
    combine_svgs_horizontal(row2_de, panel_f, row2)

    # Combine rows vertically
    combine_svgs_vertical(row1, row2, combined)

    # Add panel labels
    pw = 460.8  # panel width
    ph = 345.6  # panel height
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=2*pw+10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=pw+10, y=ph+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'F', x=2*pw+10, y=ph+20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate and panel files
    if cleanup:
        panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]
        intermediate_files = [row1_ab, row1, row2_de, row2, combined, labeled]
        for f in panel_files + intermediate_files:
            if os.path.exists(f): os.remove(f)