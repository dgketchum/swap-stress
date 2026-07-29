"""Figure 1: how the dataset is built.

A schematic of the prediction chain: static landscape covariates and daily
SMAP L3 soil moisture enter a random forest trained on harmonised (theta,
suction) pairs, which emits daily CONUS suction maps. Feature groups reflect
the global-pruned ablation (sentinel-1, SMAP climatology and land cover were
dropped at threshold r2_drop <= 0).

The schematic is drawn in matplotlib rather than hand-written SVG so that it
inherits ``style`` -- the same typeface, the same 7 pt ceiling and the same
RGB palette as the other five descriptor figures -- and so that it can be
written as a PDF with the text still text (``pdf.fonttype`` 42). Nature accepts
PDF/EPS vector artwork and asks that line art and text never be rasterised; the
previous SVG used ``feDropShadow``, which any SVG-to-PDF step flattens to a
bitmap, so the shadows are gone.

Geometry is in millimetres. A single full-bleed axes spans the figure and its
data limits are the figure's millimetre extent, so every constant below is a
true printed dimension. That also means no ``bbox_inches="tight"`` and no
layout engine is needed: nothing can fall outside the frame, so the declared
183 mm width survives to disk.

Outputs PDF, PNG and SVG to --output-dir.

Usage:
    uv run swapstress-figures --figure pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from swapstress.figures import style

# ---------------------------------------------------------------------------
# Canvas -- double column, comfortably inside the 247 mm page depth
# ---------------------------------------------------------------------------
FIG_W_MM = style.DOUBLE_COLUMN_MM
FIG_H_MM = 90.0

MARGIN_MM = 3.5

# Three columns: covariate stack | model spine | product
COL_L_X, COL_L_W = 3.0, 52.0
COL_C_X, COL_C_W = 67.0, 49.0
COL_R_X, COL_R_W = 128.0, 52.0

CARD_H_MM = 22.0
CARD_ROUND_MM = 1.4
MID_Y = FIG_H_MM / 2.0

# ---------------------------------------------------------------------------
# Type -- 7 pt is the guide's hard ceiling, 5 pt its hard floor
# ---------------------------------------------------------------------------
TITLE_PT = style.MAX_TEXT_PT  # 7
BODY_PT = 6.0
NOTE_PT = 5.5
LEADING = 1.55  # multiple of the type size

INK = "#1a1a1a"
BODY_INK = "#3a3a3a"
ARROW_INK = "#555555"

# Colour marks the role, not the box: the three data streams that feed the
# model each keep one hue from the validated categorical palette; the model and
# the product it emits are neutral. All type stays dark on a near-white fill,
# so nothing depends on reversed small type surviving the press.
COVARIATE, SMAP, TRAINING = style.CATEGORICAL  # blue, orange, purple
NEUTRAL = "#3f4a55"

# Feature groups ordered by ablation R^2 drop (global model):
# soilgrids +0.021, landsat +0.007, worldclim +0.006, fao +0.002, et0 not ablated
GROUPS = [
    ("SoilGrids", 20),
    ("Landsat", 70),
    ("WorldClim", 16),
    ("FAO / HWSD", 15),
    (r"Global $\mathregular{ET_o}$", 6),
]

TOTAL_FEATURES = 130  # 127 from groups + theta + depth_cm + rosetta_level

# Training data
TRAIN_OBS = "193K"
TRAIN_SITES = "2,723"
TRAIN_PROFILES = "3,300"
TRAIN_SAMPLES = "13,361"
TRAIN_SOURCES = 5

# Output
VALID_PIXELS = "~121,000"

ROW_H_MM = 6.2
ROW_GAP_MM = 1.1
PAD_MM = 2.6


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _tint(color: str, frac: float) -> tuple[float, float, float]:
    """``color`` laid over white at ``frac`` strength."""
    r, g, b = to_rgb(color)
    return tuple(1.0 - frac * (1.0 - c) for c in (r, g, b))


def _mm(pt: float) -> float:
    """Type size in points to millimetres."""
    return pt / 72.0 * 25.4


def _card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    fill: float = 0.10,
    lw: float = 0.6,
) -> None:
    """A rounded panel: accent-tinted fill, accent hairline edge."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={CARD_ROUND_MM}",
            mutation_scale=1.0,
            facecolor=_tint(accent, fill),
            edgecolor=accent,
            linewidth=lw,
            zorder=2,
        )
    )


def _stack(ax, cx: float, top: float, lines) -> float:
    """Centre a block of ``(text, size_pt, weight, colour)`` lines under ``top``.

    Returns the bottom of the block. Text is placed with ``va="top"`` and the
    cursor advances by the type size times the leading, so line spacing is set
    in printed millimetres rather than left to the renderer.
    """
    cursor = top
    for text, size, weight, color in lines:
        ax.text(
            cx,
            cursor,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            ha="center",
            va="top",
            zorder=4,
        )
        cursor -= _mm(size) * LEADING
    return cursor


def _block_height(lines) -> float:
    return sum(_mm(size) * LEADING for _, size, _, _ in lines)


def _panel(ax, x: float, y: float, w: float, accent: str, lines, fill=0.10, lw=0.6):
    """A card of the standard height with its text block vertically centred."""
    _card(ax, x, y, w, CARD_H_MM, accent, fill=fill, lw=lw)
    block = _block_height(lines)
    _stack(ax, x + w / 2.0, y + (CARD_H_MM + block) / 2.0, lines)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=5.0,
            linewidth=0.8,
            color=ARROW_INK,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            zorder=1,
        )
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _covariate_stack(ax) -> None:
    """Left panel: the static covariate stack, one row per feature group.

    Each row carries a bar scaled to its feature count, so the fact that
    Landsat dominates the stack is visible as well as stated.
    """
    header = [
        ("Static landscape covariates", TITLE_PT, "bold", INK),
        (f"{TOTAL_FEATURES} features, 9 km grid", BODY_PT, "normal", BODY_INK),
    ]
    rows_h = len(GROUPS) * ROW_H_MM + (len(GROUPS) - 1) * ROW_GAP_MM
    h = PAD_MM + _block_height(header) + 1.4 + rows_h + PAD_MM
    y = MID_Y - h / 2.0

    _card(ax, COL_L_X, y, COL_L_W, h, COVARIATE)
    cursor = _stack(ax, COL_L_X + COL_L_W / 2.0, y + h - PAD_MM, header) - 1.4

    row_x = COL_L_X + 2.4
    row_w = COL_L_W - 4.8
    widest = max(n for _, n in GROUPS)
    for label, n_feat in GROUPS:
        row_y = cursor - ROW_H_MM
        ax.add_patch(
            Rectangle(
                (row_x, row_y),
                row_w,
                ROW_H_MM,
                facecolor="white",
                edgecolor="none",
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (row_x, row_y),
                row_w * n_feat / widest,
                ROW_H_MM,
                facecolor=_tint(COVARIATE, 0.34),
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            row_x + 1.4,
            row_y + ROW_H_MM / 2.0,
            label,
            fontsize=BODY_PT,
            color=INK,
            ha="left",
            va="center",
            zorder=4,
        )
        ax.text(
            row_x + row_w - 1.4,
            row_y + ROW_H_MM / 2.0,
            str(n_feat),
            fontsize=BODY_PT,
            color=INK,
            ha="right",
            va="center",
            zorder=4,
        )
        cursor = row_y - ROW_GAP_MM


def build_figure():
    """Assemble the schematic and return the figure."""
    style.apply()
    fig = plt.figure(figsize=style.figsize(FIG_W_MM, FIG_H_MM))
    # Full-bleed axes with millimetre data limits: one data unit is one
    # millimetre on the printed page, in both directions.
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, FIG_W_MM)
    ax.set_ylim(0.0, FIG_H_MM)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    _covariate_stack(ax)

    smap_y = FIG_H_MM - MARGIN_MM - CARD_H_MM
    _panel(
        ax,
        COL_C_X,
        smap_y,
        COL_C_W,
        SMAP,
        [
            ("SMAP L3", TITLE_PT, "bold", INK),
            ("Daily soil moisture (θ)", BODY_PT, "normal", BODY_INK),
            ("AM pass, 9 km EASE-Grid 2", BODY_PT, "normal", BODY_INK),
        ],
    )

    train_y = MARGIN_MM
    _panel(
        ax,
        COL_C_X,
        train_y,
        COL_C_W,
        TRAINING,
        [
            ("Training data", TITLE_PT, "bold", INK),
            (f"{TRAIN_OBS} θ–ψ observations", BODY_PT, "normal", BODY_INK),
            (
                f"{TRAIN_SITES} sites / {TRAIN_PROFILES} profiles",
                BODY_PT,
                "normal",
                BODY_INK,
            ),
            (f"{TRAIN_SAMPLES} depth-samples", BODY_PT, "normal", BODY_INK),
            (f"{TRAIN_SOURCES} sources", NOTE_PT, "normal", BODY_INK),
        ],
    )

    rf_y = MID_Y - CARD_H_MM / 2.0
    _panel(
        ax,
        COL_C_X,
        rf_y,
        COL_C_W,
        NEUTRAL,
        [
            ("Random forest", TITLE_PT, "bold", INK),
            ("250 trees", BODY_PT, "normal", BODY_INK),
            ("Median imputation", BODY_PT, "normal", BODY_INK),
            ("5-fold MGRS spatial CV", BODY_PT, "normal", BODY_INK),
        ],
    )

    _panel(
        ax,
        COL_R_X,
        rf_y,
        COL_R_W,
        NEUTRAL,
        [
            ("Daily CONUS maps", TITLE_PT, "bold", INK),
            (
                r"$\mathregular{log_{10}}$ suction (cm $\mathregular{H_2O}$)",
                BODY_PT,
                "normal",
                BODY_INK,
            ),
            ("9 km, 2015–present", BODY_PT, "normal", BODY_INK),
            (f"{VALID_PIXELS} land pixels", BODY_PT, "normal", BODY_INK),
            ("Gap-filled along time axis", NOTE_PT, "normal", BODY_INK),
        ],
        # The released product is the terminus: same neutral, a shade stronger.
        fill=0.13,
        lw=0.9,
    )

    # Flow: covariates and the model spine converge on the random forest, which
    # emits the product.
    spine_x = COL_C_X + COL_C_W / 2.0
    _arrow(ax, COL_L_X + COL_L_W, MID_Y, COL_C_X, MID_Y)
    _arrow(ax, spine_x, smap_y, spine_x, rf_y + CARD_H_MM)
    _arrow(ax, spine_x, train_y + CARD_H_MM, spine_x, rf_y)
    _arrow(ax, COL_C_X + COL_C_W, MID_Y, COL_R_X, MID_Y)

    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate Fig 1: pipeline schematic")
    parser.add_argument(
        "--output-dir",
        default="figs/descriptor",
        help="Output directory for the rendered figure",
    )
    args = parser.parse_args(argv)

    fig = build_figure()
    png = style.save(
        fig, Path(args.output_dir) / "fig01_pipeline", ("pdf", "png", "svg")
    )
    print(f"Wrote {png}")
    for ext in ("pdf", "svg"):
        print(f"Wrote {png.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
