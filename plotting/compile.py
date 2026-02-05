# Functions to compile multi-panel figures from individual SVG panels.
from configs import *
from plotting.svgtools import *

import os


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
