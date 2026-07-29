"""Nature Scientific Data figure specification, in one place.

Every descriptor figure imports this so the set is typographically identical.
The numbers are from Nature's *Guide to Preparing Final Artwork*:

    Widths      89 mm (1 column), 120 or 136 mm (1.5), 183 mm (2 column)
    Max depth   247 mm (full page)
    Panel labels  8 pt bold, upright, lowercase -- a, b, c
    Other text  7 pt maximum, 5 pt minimum
    Typeface    sans-serif, preferably Helvetica or Arial
    Colour      RGB
    Rasterising "Do not rasterize line art or text in submitted figures"

Two consequences that are easy to get wrong:

``bbox_inches="tight"`` must not be used. It crops to the drawn content, so the
saved file is whatever width the content happened to need -- not the 183 mm the
figure declared. Nature typesets at a fixed column width, so the declared size
has to survive to disk. ``save`` therefore writes at the true figure size and
figures should use ``layout="constrained"`` to fit content inside it instead.

``pdf.fonttype`` is 42, embedding TrueType so text stays selectable and
editable. The default (Type 3) and any "convert text to outlines" step are both
what the guide explicitly rejects. Heavy data layers -- scatter clouds, mesh --
should still pass ``rasterized=True``: that rasterises the *marks* while leaving
axes, ticks and labels as vector, which is what the rule actually asks for.

The categorical palette is validated, not chosen by eye. Against a light
surface it passes lightness band, chroma floor, CVD separation (worst adjacent
pair dE 22.0 tritan / 22.7 protan), normal-vision separation (dE 28.6) and 3:1
contrast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4

SINGLE_COLUMN_MM = 89.0
NARROW_WIDE_MM = 120.0
WIDE_MM = 136.0
DOUBLE_COLUMN_MM = 183.0
MAX_DEPTH_MM = 247.0

PANEL_LABEL_PT = 8.0
MAX_TEXT_PT = 7.0
MIN_TEXT_PT = 5.0

# Helvetica and Arial first as the guide asks. Liberation Sans (Arial metrics)
# precedes Nimbus Sans (Helvetica metrics) because it is a real TrueType file:
# with ``pdf.fonttype`` 42 the PDF declares TrueType, and Nimbus Sans is
# OTF/CFF, so pdffonts reports "Mismatch between font type and embedded font
# file". Readers handle it, but a production house may not, and the guide
# accepts either face -- so take the one that does not provoke the warning.
SANS_STACK = [
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "Nimbus Sans",
    "DejaVu Sans",
]

# Fixed order -- a series keeps its colour no matter how many are drawn.
CATEGORICAL = ("#2166AC", "#D55E00", "#7B3294")

# Perceptually uniform and CVD-safe. Sequential for magnitude; diverging only
# where zero or a midpoint means something, with its neutral in the middle.
SEQUENTIAL = "cividis"
DIVERGING = "RdBu_r"

GRID_COLOR = "#d9d9d9"
AXIS_COLOR = "#333333"
MUTED_INK = "#666666"


def mm(*values: float) -> tuple[float, ...]:
    """Millimetres to inches, for figsize."""
    return tuple(v / MM_PER_INCH for v in values)


def figsize(width_mm: float, height_mm: float) -> tuple[float, float]:
    """A figure size in inches, refusing anything taller than the page."""
    if height_mm > MAX_DEPTH_MM:
        raise ValueError(
            f"{height_mm:.0f} mm exceeds the {MAX_DEPTH_MM:.0f} mm page depth"
        )
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def resolved_family() -> str:
    """The first face in ``SANS_STACK`` actually installed on this machine."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in SANS_STACK if name in installed), "DejaVu Sans")


def mathtext_params() -> dict:
    """Bind mathtext to the same face the body text resolves to.

    Left alone, matplotlib draws ``$\\log_{10}$`` and ``R$^2$`` from its own
    font set -- DejaVu -- no matter what ``font.sans-serif`` says. The PDF then
    embeds two families and the artwork is set in two typefaces, which is
    invisible on screen and a spec violation in print. These keys need a
    concrete family name, since the generic ``sans-serif`` is not a legal
    fontconfig pattern, so the stack is resolved against what is installed.

    Note this is not avoidable by writing Unicode ``log₁₀`` instead: the
    Helvetica/Arial clones have no subscript-digit glyphs, so that falls back
    to DejaVu too.
    """
    family = resolved_family()
    return {
        "mathtext.fontset": "custom",
        "mathtext.default": "regular",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.tt": family,
        "mathtext.cal": f"{family}:italic",
    }


def apply() -> None:
    """Install the specification as matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SANS_STACK,
            **mathtext_params(),
            "font.size": MAX_TEXT_PT,
            "axes.titlesize": MAX_TEXT_PT,
            "axes.labelsize": MAX_TEXT_PT,
            "xtick.labelsize": MAX_TEXT_PT - 1,
            "ytick.labelsize": MAX_TEXT_PT - 1,
            "legend.fontsize": MAX_TEXT_PT - 1,
            "figure.titlesize": MAX_TEXT_PT,
            # Text stays text, in every vector format we emit.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "axes.linewidth": 0.5,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": AXIS_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
            "axes.grid": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
        }
    )


def panel_label(ax, letter: str, dx: float = -0.06, dy: float = 1.04) -> None:
    """Label a panel of a multi-part figure: 8 pt bold, upright, lowercase."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_PT,
        fontweight="bold",
        fontstyle="normal",
        va="bottom",
        ha="left",
    )


def save(fig, stem: str | Path, formats: Iterable[str] = ("pdf", "png")) -> Path:
    """Write a figure at exactly its declared size and return the PNG path.

    No ``bbox_inches="tight"`` -- see the module docstring. Use
    ``layout="constrained"`` on the figure so nothing needs cropping.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return next((p for p in written if p.suffix == ".png"), written[0]).absolute()
