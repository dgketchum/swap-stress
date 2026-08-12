"""Descriptor Fig 1: how the dataset is built.

A schematic of the prediction chain: static landscape covariates and daily
SMAP L3 soil moisture enter a quantile random forest trained on harmonised
(theta, suction) pairs, which emits a median and a 95% prediction interval on
the days SMAP retrieved -- Level 1 -- and, after a temporal gap-fill, a value
for every calendar day -- Level 2. Matric potential in signed MPa is the only
unit the released files carry (the model-native log10 suction is internal), so
the cards name the released bands: the median plus its q025/q975 pair at
Level 1, the median plus ``gapfill_flag`` at Level 2 -- the interval ships at
Level 1 only, because on a filled day the model never ran.
Feature groups reflect the global-pruned ablation (sentinel-1, SMAP climatology
and land cover were dropped at threshold r2_drop <= 0). The covariate card
counts only the 127 static features; theta (the SMAP card) and the two fixed
sample descriptors depth_cm and rosetta_level -- real model features held at
constant values for the product run -- are named on the model card, so the
inputs shown sum to the model's 130 features.

Gap-fill is drawn as a stage of its own rather than folded into a line of the
product card. The Level 1 / Level 2 split is what Figs 2 and 3 are about -- Fig
3 maps where Level 1 is sparse; the pixel-series supporting panel draws Level 1 over the Level 2
line -- so the schematic has to show the two as separate things a reuser can
download, with the rule that turns one into the other named in between.

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
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from swapstress.figures import style

# ---------------------------------------------------------------------------
# Canvas -- double column, comfortably inside the 247 mm page depth
# ---------------------------------------------------------------------------
FIG_W_MM = style.DOUBLE_COLUMN_MM
FIG_H_MM = 90.0

MARGIN_MM = 3.5

# Three columns: covariate stack | model spine | product ladder
COL_L_X, COL_L_W = 3.0, 52.0
COL_C_X, COL_C_W = 67.0, 49.0
COL_R_X, COL_R_W = 128.0, 52.0

CARD_H_MM = 22.0
# The gap-fill stage is a rule applied to the level above it, not a product a
# reuser downloads, so it is a shorter band between the two full-height cards.
BAND_H_MM = 15.0
# Vertical run between stacked stages. Set to the 12 mm gutter between the
# columns, which also lands the ladder on the spine's grid: Level 1 tops out
# with SMAP, the gap-fill band sits at the model's mid-height, and Level 2
# bottoms out with the training card.
STACK_GAP_MM = 12.0
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

# The covariate card shows only the static stack; theta (SMAP card) and the two
# fixed sample descriptors (model card) bring the model's input count to 130.
STATIC_FEATURES = sum(n for _, n in GROUPS)  # 127
MODEL_INPUTS = STATIC_FEATURES + 3  # + theta + depth_cm + rosetta_level

# Training data (profile / depth-sample counts live in the caption)
TRAIN_OBS = "193K"
TRAIN_LOCATIONS = "2,607"
TRAIN_SOURCES = 5

# Output
VALID_PIXELS = "119,693"

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


def _panel(
    ax,
    x: float,
    y: float,
    w: float,
    accent: str,
    lines,
    fill=0.10,
    lw=0.6,
    h: float = CARD_H_MM,
):
    """A card of height *h* with its text block vertically centred."""
    _card(ax, x, y, w, h, accent, fill=fill, lw=lw)
    block = _block_height(lines)
    _stack(ax, x + w / 2.0, y + (h + block) / 2.0, lines)


_ARROW_KW = dict(
    arrowstyle="-|>",
    mutation_scale=5.0,
    linewidth=0.8,
    color=ARROW_INK,
    shrinkA=0,
    shrinkB=0,
    joinstyle="miter",
    zorder=1,
)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **_ARROW_KW))


def _elbow(ax, points) -> None:
    """A right-angled arrow through *points*, head on the last.

    The model sits at mid-height but the product ladder is a stack, so the arrow
    into its top card has to rise as well as run. Two bends rather than a
    diagonal: every other connector here is orthogonal, and a lone slanted line
    would read as a different kind of relation.
    """
    ax.add_patch(FancyArrowPatch(path=MplPath(points), **_ARROW_KW))


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
        (f"{STATIC_FEATURES} features, 9 km grid", BODY_PT, "normal", BODY_INK),
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
            (f"{TRAIN_LOCATIONS} unique locations", BODY_PT, "normal", BODY_INK),
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
            ("Quantile random forest", TITLE_PT, "bold", INK),
            (
                f"{MODEL_INPUTS} inputs: {STATIC_FEATURES} covariates + θ",
                BODY_PT,
                "normal",
                BODY_INK,
            ),
            ("+ depth & Rosetta level (fixed)", BODY_PT, "normal", BODY_INK),
            ("9 km spatial-group holdout", NOTE_PT, "normal", BODY_INK),
        ],
    )

    # Product ladder, centred on the model so the column reads as one block:
    # Level 1, the rule that fills it, then Level 2. Both levels are released,
    # so both carry the stronger fill; the gap-fill band between them is lighter
    # because it is a step, not something a reuser downloads.
    stack_h = 2 * CARD_H_MM + BAND_H_MM + 2 * STACK_GAP_MM
    l1_y = MID_Y + stack_h / 2.0 - CARD_H_MM
    fill_y = l1_y - STACK_GAP_MM - BAND_H_MM
    l2_y = fill_y - STACK_GAP_MM - CARD_H_MM

    _panel(
        ax,
        COL_R_X,
        l1_y,
        COL_R_W,
        NEUTRAL,
        [
            ("Level 1", TITLE_PT, "bold", INK),
            ("QRF median ψ + q025 / q975 (MPa)", BODY_PT, "normal", BODY_INK),
            ("9 km, 2015–present", BODY_PT, "normal", BODY_INK),
            (f"{VALID_PIXELS} land pixels", BODY_PT, "normal", BODY_INK),
            ("Retrieval days only", NOTE_PT, "normal", BODY_INK),
        ],
        fill=0.13,
        lw=0.9,
    )

    _panel(
        ax,
        COL_R_X,
        fill_y,
        COL_R_W,
        NEUTRAL,
        [
            ("Temporal gap-fill", TITLE_PT, "bold", INK),
            ("Linear interpolation per pixel", BODY_PT, "normal", BODY_INK),
            ("Ends held flat, not observed", NOTE_PT, "normal", BODY_INK),
        ],
        fill=0.05,
        lw=0.6,
        h=BAND_H_MM,
    )

    _panel(
        ax,
        COL_R_X,
        l2_y,
        COL_R_W,
        NEUTRAL,
        [
            ("Level 2", TITLE_PT, "bold", INK),
            ("Gap-filled daily median ψ (MPa)", BODY_PT, "normal", BODY_INK),
            ("median + gapfill_flag; interval in L1", BODY_PT, "normal", BODY_INK),
            ("Same grid, every calendar day", NOTE_PT, "normal", BODY_INK),
        ],
        fill=0.13,
        lw=0.9,
    )

    # Flow: covariates and the model spine converge on the random forest, which
    # emits Level 1; the ladder then runs down to Level 2.
    spine_x = COL_C_X + COL_C_W / 2.0
    ladder_x = COL_R_X + COL_R_W / 2.0
    turn_x = (COL_C_X + COL_C_W + COL_R_X) / 2.0
    _arrow(ax, COL_L_X + COL_L_W, MID_Y, COL_C_X, MID_Y)
    _arrow(ax, spine_x, smap_y, spine_x, rf_y + CARD_H_MM)
    _arrow(ax, spine_x, train_y + CARD_H_MM, spine_x, rf_y)
    _elbow(
        ax,
        [
            (COL_C_X + COL_C_W, MID_Y),
            (turn_x, MID_Y),
            (turn_x, l1_y + CARD_H_MM / 2.0),
            (COL_R_X, l1_y + CARD_H_MM / 2.0),
        ],
    )
    _arrow(ax, ladder_x, l1_y, ladder_x, fill_y + BAND_H_MM)
    _arrow(ax, ladder_x, fill_y, ladder_x, l2_y + CARD_H_MM)

    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Descriptor Fig 1: pipeline schematic")
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
