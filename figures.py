"""Functions to generate main manuscript figures."""

from configs import *
from subjects import *
from analysis import *
from plots import *
from svgtools import *

import os


# Default normative model parameters
NORMATIVE_PARAMS = {
    'init_state_est': 150,      # Initial state estimate (middle of range)
    'init_runlen_est': 1,       # Initial run length estimate
    'noise_sd': 25.0,           # Observation noise SD (from task)
    'hazard': 0.1,              # Hazard rate (from task)
    'drift': 0.0,               # No drift in normative model
    'noise_sd_update': 0.0,     # No update noise in normative model
    'limit_updates': False,     # Don't limit updates
    'clip': True,               # Clip final runlen
}

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

    # Use PC scores to find most and least adaptive subjects
    # Score_0 captures overall learning rate level
    # Score_1 captures adaptivity (high = more adaptive around changepoints)
    scores = subj_pca_scores['Score_1'].values

    # Non-adaptive: lowest Score_1 (flat learning rate profile)
    snum_nonadapt = np.argmin(scores)

    # Adaptive: highest Score_1 (peaked learning rate at CP)
    snum_adapt = np.argmax(scores)

    print(f"Non-adaptive subject: {snum_nonadapt} (Score_1 = {scores[snum_nonadapt]:.3f})")
    print(f"Adaptive subject: {snum_adapt} (Score_1 = {scores[snum_adapt]:.3f})")

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
    params = NORMATIVE_PARAMS.copy()
    params['noise_sd'] = tasks[0].noise_sd[0]
    params['hazard'] = tasks[0].hazard[0]
    simulate_subject(sim, tasks[0].obs, params)

    ntrials = 85

    # --- Panel A: Normative predictions ---
    fig, ax = plt.subplots()
    ax.plot(tasks[0].obs[:ntrials], '.', label='Obs')
    ax.plot(sim.responses.pred[:ntrials], '.', label='Pred')
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
    beta_cols = ['m4_beta_pe', 'm4_beta_cppd', 'm4_beta_rud', 'm4_beta_prodd']

    # Check which beta columns exist
    beta_cols = [c for c in beta_cols if c in subj_linear_models.columns]

    # Combine into single dataframe for correlation
    corr_data = pd.concat([
        subj_pca_lr_scores[score_cols],
        subj_linear_models[beta_cols]
    ], axis=1)

    # Rename for display
    corr_data.columns = ['PC1', 'PC2'] + [c.replace('m4_beta_', '').upper() for c in beta_cols]

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
        Row 1: [A] PE vs Update comparison, [B] Beta strip plots, [C] Trialwise regression VE
        Row 2: [D] Cumulative VE by PCA, [E] Score strip plots, [F] PCA reconstruction VE

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

    # --- Panel A: PE vs Update comparison (normative vs PE-only) ---
    # Simulate normative and PE-only models on first task
    sim_norm = Subject()
    params_norm = NORMATIVE_PARAMS.copy()
    params_norm['noise_sd'] = tasks[0].noise_sd[0]
    params_norm['hazard'] = tasks[0].hazard[0]
    simulate_subject(sim_norm, tasks[0].obs, params_norm)

    sim_pe = Subject()
    params_pe = NORMATIVE_PARAMS.copy()
    params_pe['noise_sd'] = tasks[0].noise_sd[0]
    params_pe['hazard'] = 0.0  # No changepoint detection
    simulate_subject(sim_pe, tasks[0].obs, params_pe)

    fig, ax = plt.subplots()
    plot_pe_update_model_comparison(sim_norm, sim_pe, ax=ax)
    ax.set_title('Normative vs PE-Only')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_A' + FIG_FMT, dpi=300)

    # --- Panel B: Beta strip plots ---
    fig, ax = plt.subplots()
    plot_subj_betas_strip(subj_linear_models, model='m4', ax=ax)
    ax.set_title('Subject Beta Coefficients')
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_B' + FIG_FMT, dpi=300)

    # --- Panel C: Trialwise regression VE ---
    fig, ax = plt.subplots()
    plot_lm_ve(subjs, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_C' + FIG_FMT, dpi=300)

    # --- Panel D: Cumulative VE by PCA ---
    fig, ax = plt.subplots()
    plot_cumulative_ve(cumulative_ve, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_D' + FIG_FMT, dpi=300)

    # --- Panel E: Score strip plots ---
    fig, ax = plt.subplots()
    plot_scores_strip(subj_pca_scores, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_E' + FIG_FMT, dpi=300)

    # --- Panel F: PCA reconstruction VE ---
    reconstruction_ve = get_subj_pca_ve(subj_pcp_lr, group_pca_basis, subj_pca_scores)
    fig, ax = plt.subplots()
    plot_pca_reconstruction_ve(reconstruction_ve, ax=ax)
    fig.tight_layout()
    if savefig: fig.savefig(FIGURES_DIR + 'fig3_F' + FIG_FMT, dpi=300)

    if close: plt.close('all')


def compile_figure_1(cleanup=True):
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

    # Add panel labels (positions will need adjustment based on actual panel sizes)
    add_text_to_svg(combined, labeled, 'C', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=160, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=310, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'F', x=10, y=160, font_size=14)
    add_text_to_svg(labeled, labeled, 'G', x=160, y=160, font_size=14)
    add_text_to_svg(labeled, labeled, 'H', x=310, y=160, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate files
    if cleanup:
        for f in [row1_cd, row1, row2_fg, row2, combined, labeled]:
            if os.path.exists(f): os.remove(f)


def compile_figure_2(cleanup=True):
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
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=160, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=160, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=160, y=160, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=310, y=20, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate files
    if cleanup:
        for f in [col1, col2, cols_12, combined, labeled]:
            if os.path.exists(f): os.remove(f)


def compile_figure_3(cleanup=True):
    """Compile Figure 3 from individual panels.

    Layout:
        Row 1: [A] [B] [C]  (PE comparison, beta strips, regression VE)
        Row 2: [D] [E] [F]  (cumulative VE, score strips, PCA reconstruction VE)
    """
    # Input panel files
    panel_a = FIGURES_DIR + 'fig3_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig3_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig3_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig3_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig3_E' + FIG_FMT
    panel_f = FIGURES_DIR + 'fig3_F' + FIG_FMT

    # Intermediate files
    row1_ab = FIGURES_DIR + 'fig3_row1_ab.svg'
    row1 = FIGURES_DIR + 'fig3_row1.svg'
    row2_de = FIGURES_DIR + 'fig3_row2_de.svg'
    row2 = FIGURES_DIR + 'fig3_row2.svg'
    combined = FIGURES_DIR + 'fig3_combined.svg'
    labeled = FIGURES_DIR + 'fig3_labeled.svg'
    final = FIGURES_DIR + 'fig3_final.svg'

    # Build row 1: A + B + C
    combine_svgs_horizontal(panel_a, panel_b, row1_ab)
    combine_svgs_horizontal(row1_ab, panel_c, row1)

    # Build row 2: D + E + F
    combine_svgs_horizontal(panel_d, panel_e, row2_de)
    combine_svgs_horizontal(row2_de, panel_f, row2)

    # Combine rows
    combine_svgs_vertical(row1, row2, combined)

    # Add panel labels
    add_text_to_svg(combined, labeled, 'A', x=10, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'B', x=220, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'C', x=430, y=20, font_size=14)
    add_text_to_svg(labeled, labeled, 'D', x=10, y=170, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=220, y=170, font_size=14)
    add_text_to_svg(labeled, labeled, 'F', x=430, y=170, font_size=14)

    # Scale to final width
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    # Convert to PDF
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    # Clean up intermediate files
    if cleanup:
        for f in [row1_ab, row1, row2_de, row2, combined, labeled]:
            if os.path.exists(f): os.remove(f)