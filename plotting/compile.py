# Functions to compile multi-panel figures from individual SVG panels.
from configs import *
from plotting.svgtools import *

import os

# Standard panel dimensions (6.4 x 4.8 inches at 72 dpi)
PANEL_WIDTH = 460.8
PANEL_HEIGHT = 345.6


def _combine_row(panels, output):
    """Combine a list of panel files horizontally into output."""
    if len(panels) == 1:
        import shutil
        shutil.copy(panels[0], output)
        return

    # Combine first two
    tmp = output.replace('.svg', '_tmp.svg')
    combine_svgs_horizontal(panels[0], panels[1], tmp if len(panels) > 2 else output)

    # Add remaining panels
    for i, panel in enumerate(panels[2:], start=2):
        src = tmp
        dst = output if i == len(panels) - 1 else tmp.replace('_tmp', f'_tmp{i}')
        combine_svgs_horizontal(src, panel, dst)
        if os.path.exists(src) and src != dst:
            os.remove(src)
        tmp = dst


def _combine_grid(rows, output):
    """Combine a list of row files vertically into output."""
    if len(rows) == 1:
        import shutil
        shutil.copy(rows[0], output)
        return

    # Combine first two
    tmp = output.replace('.svg', '_tmp.svg')
    combine_svgs_vertical(rows[0], rows[1], tmp if len(rows) > 2 else output)

    # Add remaining rows
    for i, row in enumerate(rows[2:], start=2):
        src = tmp
        dst = output if i == len(rows) - 1 else tmp.replace('_tmp', f'_tmp{i}')
        combine_svgs_vertical(src, row, dst)
        if os.path.exists(src) and src != dst:
            os.remove(src)
        tmp = dst


def _add_grid_labels(svg_in, svg_out, labels, col_widths=None, row_heights=None):
    """Add panel labels at grid positions.

    Parameters:
        svg_in (str)       - Input SVG path.
        svg_out (str)      - Output SVG path.
        labels (list)      - 2D list of labels, row-major order. None entries are skipped.
        col_widths (list)  - Width of each column. Defaults to PANEL_WIDTH for all.
        row_heights (list) - Height of each row. Defaults to PANEL_HEIGHT for all.
    """
    n_rows = len(labels)
    n_cols = max(len(row) for row in labels)

    if col_widths is None:
        col_widths = [PANEL_WIDTH] * n_cols
    if row_heights is None:
        row_heights = [PANEL_HEIGHT] * n_rows

    current = svg_in
    for r, row in enumerate(labels):
        y = sum(row_heights[:r]) + 20
        for c, label in enumerate(row):
            if label is None:
                continue
            x = sum(col_widths[:c]) + 10
            add_text_to_svg(current, svg_out, label, x=x, y=y, font_size=14)
            current = svg_out


def _finalize(labeled, final, panel_files, intermediate_files, cleanup):
    """Scale to final width, convert to PDF, and optionally clean up."""
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)
    svg_to_pdf(final, final.replace('.svg', '.pdf'))

    if cleanup:
        for f in panel_files + intermediate_files:
            if os.path.exists(f):
                os.remove(f)


def _compile_grid(fig_num, layout, labels, cleanup=FIG_CLEANUP, col_widths=None, row_heights=None):
    """Generic grid compilation.

    Parameters:
        fig_num (int)      - Figure number.
        layout (list)      - 2D list of panel labels (e.g., [['A','B','C'], ['D','E','F']]).
        labels (list)      - 2D list of display labels (can differ from layout for custom labeling).
        cleanup (bool)     - Whether to clean up intermediate files.
        col_widths (list)  - Width of each column.
        row_heights (list) - Height of each row.
    """
    prefix = f'fig{fig_num}'

    # Build panel file paths
    panel_files = []
    for row in layout:
        for label in row:
            if label is not None:
                panel_files.append(FIGURES_DIR + f'{prefix}_{label}' + FIG_FMT)

    # Intermediate files
    row_files = [FIGURES_DIR + f'{prefix}_row{r}.svg' for r in range(len(layout))]
    combined = FIGURES_DIR + f'{prefix}_combined.svg'
    labeled_file = FIGURES_DIR + f'{prefix}_labeled.svg'
    final = FIGURES_DIR + f'{prefix}_final.svg'

    # Build each row
    for r, row in enumerate(layout):
        row_panels = [FIGURES_DIR + f'{prefix}_{label}' + FIG_FMT for label in row if label is not None]
        _combine_row(row_panels, row_files[r])

    # Combine rows
    _combine_grid(row_files, combined)

    # Add labels
    _add_grid_labels(combined, labeled_file, labels, col_widths, row_heights)

    # Finalize
    intermediate = row_files + [combined, labeled_file]
    _finalize(labeled_file, final, panel_files, intermediate, cleanup)


def compile_figure_1(cleanup=FIG_CLEANUP):
    """Compile Figure 1: two rows of three panels each."""
    _compile_grid(
        fig_num=1,
        layout=[['C', 'D', 'E'], ['F', 'G', 'H']],
        labels=[['C', 'D', 'E'], ['F', 'G', 'H']],
        cleanup=cleanup
    )


def compile_figure_2(cleanup=FIG_CLEANUP):
    """Compile Figure 2: columns A/D, B/E stacked, then C alongside.

    This layout is non-standard, so we handle it explicitly.
    """
    panel_a = FIGURES_DIR + 'fig2_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig2_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig2_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig2_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig2_E' + FIG_FMT

    col1 = FIGURES_DIR + 'fig2_col1.svg'
    col2 = FIGURES_DIR + 'fig2_col2.svg'
    cols_12 = FIGURES_DIR + 'fig2_cols12.svg'
    combined = FIGURES_DIR + 'fig2_combined.svg'
    labeled = FIGURES_DIR + 'fig2_labeled.svg'
    final = FIGURES_DIR + 'fig2_final.svg'

    # Build columns
    combine_svgs_vertical(panel_a, panel_d, col1)
    combine_svgs_vertical(panel_b, panel_e, col2)

    # Merge columns horizontally
    combine_svgs_horizontal(col1, col2, cols_12)
    combine_svgs_horizontal(cols_12, panel_c, combined)

    # Add labels
    _add_grid_labels(combined, labeled,
                     [['A', 'B', 'C'], ['D', 'E', None]])

    # Finalize
    panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e]
    intermediate = [col1, col2, cols_12, combined, labeled]
    _finalize(labeled, final, panel_files, intermediate, cleanup)


def compile_figure_3(cleanup=FIG_CLEANUP):
    """Compile Figure 3: row 1 has 3 standard panels, row 2 has 2 wide panels."""
    panel_a = FIGURES_DIR + 'fig3_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig3_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig3_C' + FIG_FMT
    panel_d = FIGURES_DIR + 'fig3_D' + FIG_FMT
    panel_e = FIGURES_DIR + 'fig3_E' + FIG_FMT

    row1 = FIGURES_DIR + 'fig3_row1.svg'
    row2 = FIGURES_DIR + 'fig3_row2.svg'
    combined = FIGURES_DIR + 'fig3_combined.svg'
    labeled = FIGURES_DIR + 'fig3_labeled.svg'
    final = FIGURES_DIR + 'fig3_final.svg'

    # Build rows
    _combine_row([panel_a, panel_b, panel_c], row1)
    _combine_row([panel_d, panel_e], row2)
    _combine_grid([row1, row2], combined)

    # Row 2 has wide panels (691.2 pt each)
    wpw = 691.2
    _add_grid_labels(combined, labeled,
                     [['A', 'B', 'C'], ['D', 'E']],
                     col_widths=[PANEL_WIDTH, PANEL_WIDTH, PANEL_WIDTH],
                     row_heights=[PANEL_HEIGHT, PANEL_HEIGHT])

    # Override second row label positions manually
    add_text_to_svg(labeled, labeled, 'D', x=10, y=PANEL_HEIGHT+20, font_size=14)
    add_text_to_svg(labeled, labeled, 'E', x=wpw+10, y=PANEL_HEIGHT+20, font_size=14)

    panel_files = [panel_a, panel_b, panel_c, panel_d, panel_e]
    intermediate = [row1, row2, combined, labeled]
    _finalize(labeled, final, panel_files, intermediate, cleanup)


def compile_figure_4(cleanup=FIG_CLEANUP):
    """Compile Figure 4: single row with 2 panels."""
    _compile_grid(
        fig_num=4,
        layout=[['A', 'B']],
        labels=[['A', 'B']],
        cleanup=cleanup
    )


def compile_figure_5(cleanup=FIG_CLEANUP):
    """Compile Figure 5: 2x3 grid."""
    _compile_grid(
        fig_num=5,
        layout=[['A', 'B', 'C'], ['D', 'E', 'F']],
        labels=[['A', 'B', 'C'], ['D', 'E', 'F']],
        cleanup=cleanup
    )


def compile_figure_6(cleanup=FIG_CLEANUP):
    """Compile Figure 6: 2x3 grid."""
    _compile_grid(
        fig_num=6,
        layout=[['A', 'B', 'C'], ['D', 'E', 'F']],
        labels=[['A', 'B', 'C'], ['D', 'E', 'F']],
        cleanup=cleanup
    )


def compile_figure_7(cleanup=FIG_CLEANUP):
    """Compile Figure 7: single row with 3 panels (model comparison)."""
    _compile_grid(
        fig_num=7,
        layout=[['A', 'B', 'C']],
        labels=[['A', 'B', 'C']],
        cleanup=cleanup
    )


def compile_figure_8(cleanup=FIG_CLEANUP):
    """Compile Figure 8: row 1 has A and B, row 2 has C (wide panel for alt models)."""
    panel_a = FIGURES_DIR + 'fig8_A' + FIG_FMT
    panel_b = FIGURES_DIR + 'fig8_B' + FIG_FMT
    panel_c = FIGURES_DIR + 'fig8_C' + FIG_FMT

    row1 = FIGURES_DIR + 'fig8_row1.svg'
    combined = FIGURES_DIR + 'fig8_combined.svg'
    labeled = FIGURES_DIR + 'fig8_labeled.svg'
    final = FIGURES_DIR + 'fig8_final.svg'

    # Build row 1
    _combine_row([panel_a, panel_b], row1)
    _combine_grid([row1, panel_c], combined)

    # Add labels
    _add_grid_labels(combined, labeled,
                     [['A', 'B'], ['C', None]],
                     col_widths=[PANEL_WIDTH, PANEL_WIDTH],
                     row_heights=[PANEL_HEIGHT, PANEL_HEIGHT * 2])

    panel_files = [panel_a, panel_b, panel_c]
    intermediate = [row1, combined, labeled]
    _finalize(labeled, final, panel_files, intermediate, cleanup)


def compile_figure_9(cleanup=FIG_CLEANUP):
    """Compile Figure 9: single panel (task scatter)."""
    _compile_grid(
        fig_num=9,
        layout=[['A']],
        labels=[['A']],
        cleanup=cleanup
    )
