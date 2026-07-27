"""Figure 3: Prediction pipeline diagram.

Box-and-arrow flow: static covariates + SMAP L3 → RF → daily CONUS suction maps.
Feature groups reflect the global-pruned ablation (sentinel-1, SMAP climatology,
and land cover dropped at threshold r2_drop <= 0).

Outputs SVG and PNG to --output-dir.

Usage:
    uv run python viz/presentation/fig3_pipeline.py --output-dir figs/presentation
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
W, H = 960, 480
FONT = "Helvetica, Arial, sans-serif"

# Palette
BG = "#fafafa"
TEXT_DARK = "#2c3e50"
TEXT_MID = "#5a6d80"
ARROW = "#444"

SMAP_BG = "#b2182b"
SMAP_TEXT = "#ffcdd2"
RF_BG = "#2c3e50"
RF_TEXT = "#aab7c4"
OUTPUT_BG = "#1a7a3a"
OUTPUT_TEXT = "#c8e6c9"
OUTPUT_ACCENT = "#a5d6a7"
TRAIN_BG = "#7b5ea7"
TRAIN_TEXT = "#d7cce5"

# Feature groups ordered by ablation R² drop (global model):
# soilgrids +0.021, landsat +0.007, worldclim +0.006, fao +0.002, et0 not ablated
GROUPS = [
    ("SoilGrids", 20, "#2166ac"),
    ("Landsat", 70, "#3a7bbf"),
    ("WorldClim", 16, "#4a8fd1"),
    ("FAO / HWSD", 15, "#5da0de"),
    ("_eto_", 6, "#72b0e8"),  # rendered with SVG subscript below
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

# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _rect(
    x,
    y,
    w,
    h,
    rx=8,
    fill="#fff",
    stroke=None,
    sw=1.2,
    opacity=1.0,
    shadow=False,
    dash=False,
):
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"']
    if opacity < 1.0:
        parts.append(f' opacity="{opacity}"')
    if stroke:
        parts.append(f' stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            parts.append(' stroke-dasharray="4,3"')
    if shadow:
        parts.append(' filter="url(#shadow)"')
    parts.append("/>")
    return "".join(parts)


def _text(
    x,
    y,
    content,
    size=12,
    fill="#000",
    anchor="middle",
    weight="normal",
    style="normal",
):
    s = f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
    s += f'fill="{fill}" font-weight="{weight}" font-style="{style}">'
    s += f"{content}</text>"
    return s


def _line(x1, y1, x2, y2, stroke=ARROW, sw=2, marker="arrow"):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}" marker-end="url(#{marker})"/>'
    )


# ---------------------------------------------------------------------------
# Build SVG
# ---------------------------------------------------------------------------


def build_svg() -> str:
    lines = []
    a = lines.append

    a(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}">'
    )

    # Defs: arrow marker + drop shadow
    a("  <defs>")
    a(
        '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto">'
    )
    a(f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW}"/>')
    a("    </marker>")
    a('    <filter id="shadow" x="-4%" y="-4%" width="108%" height="108%">')
    a('      <feDropShadow dx="1.5" dy="1.5" stdDeviation="2.5" flood-opacity="0.12"/>')
    a("    </filter>")
    a("  </defs>")

    # Background
    a(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

    # ---- Static covariate stack (left) ----
    stack_x, stack_y = 55, 100
    card_w, card_h = 200, 28
    card_gap = 6
    n = len(GROUPS)
    stack_h = 60 + n * (card_h + card_gap) + 10

    a(f'  <g transform="translate({stack_x}, {stack_y})">')
    a(
        f"    {_rect(0, 0, 230, stack_h, rx=8, fill='#e8edf3', stroke='#8fa4bf', sw=1.2, shadow=True)}"
    )
    a(
        f"    {_text(115, 26, 'Static Landscape Covariates', size=13, fill=TEXT_DARK, weight='bold')}"
    )
    a(
        f"    {_text(115, 44, f'{TOTAL_FEATURES} features, 9 km resolution', size=11, fill=TEXT_MID)}"
    )

    for i, (label, n_feat, color) in enumerate(GROUPS):
        cy = 58 + i * (card_h + card_gap)
        a(f"    {_rect(15, cy, card_w, card_h, rx=4, fill=color, opacity=0.88)}")
        if label == "_eto_":
            # Separate text elements to fake subscript (cairosvg mangles tspan dy)
            a(
                f"    {_text(105, cy + 18, 'Global ET', size=11, fill='#fff', weight='bold')}"
            )
            a(
                f"    {_text(132, cy + 21, 'o', size=7.5, fill='#fff', weight='bold', anchor='start')}"
            )
            a(
                f"    {_text(139, cy + 18, '({})'.format(n_feat), size=11, fill='#fff', weight='bold', anchor='start')}"
            )
        else:
            a(
                f"    {_text(115, cy + 18, f'{label} ({n_feat})', size=11, fill='#fff', weight='bold')}"
            )

    a("  </g>")

    # Covariates → RF arrow
    arrow_y = stack_y + stack_h // 2
    a(f"  {_line(stack_x + 240, arrow_y, 388, arrow_y)}")

    # ---- SMAP L3 (top center) ----
    smap_x, smap_y = 395, 32
    a(f'  <g transform="translate({smap_x}, {smap_y})">')
    a(f"    {_rect(0, 0, 200, 80, rx=8, fill=SMAP_BG, shadow=True)}")
    a(f"    {_text(100, 28, 'SMAP L3', size=14, fill='#fff', weight='bold')}")
    theta_label = "Daily Soil Moisture (\u03b8)"
    a(f"    {_text(100, 47, theta_label, size=11.5, fill=SMAP_TEXT)}")
    a(f"    {_text(100, 63, 'AM Pass, 9 km EASE-Grid 2', size=10, fill=SMAP_TEXT)}")
    a("  </g>")

    # SMAP → RF arrow
    a(f"  {_line(smap_x + 100, smap_y + 86, smap_x + 100, 195)}")

    # ---- Random Forest (center) ----
    rf_x, rf_y = 395, 200
    a(f'  <g transform="translate({rf_x}, {rf_y})">')
    a(f"    {_rect(0, 0, 200, 110, rx=10, fill=RF_BG, shadow=True)}")
    a(f"    {_text(100, 32, 'Random Forest', size=15, fill='#fff', weight='bold')}")
    a(f"    {_text(100, 54, '250 trees', size=12, fill=RF_TEXT)}")
    a(f"    {_text(100, 74, 'Median imputation', size=11, fill=RF_TEXT)}")
    a(f"    {_text(100, 94, '5-fold MGRS spatial CV', size=11, fill=RF_TEXT)}")
    a("  </g>")

    # RF → output arrow
    a(f"  {_line(rf_x + 206, rf_y + 55, 698, rf_y + 55)}")

    # ---- Output (right) ----
    out_x, out_y = 705, 195
    a(f'  <g transform="translate({out_x}, {out_y})">')
    a(f"    {_rect(0, 0, 215, 120, rx=8, fill=OUTPUT_BG, shadow=True)}")
    a(f"    {_text(108, 28, 'Daily CONUS Maps', size=14, fill='#fff', weight='bold')}")
    log_label = "log10 suction (cm H2O)"
    a(f"    {_text(108, 50, log_label, size=12, fill=OUTPUT_TEXT)}")
    date_label = "9 km, 2015\u2013present"
    a(f"    {_text(108, 70, date_label, size=11, fill=OUTPUT_TEXT)}")
    a(f"    {_text(108, 90, f'{VALID_PIXELS} land pixels', size=11, fill=OUTPUT_TEXT)}")
    a(
        f"    {_text(108, 106, 'Gap-filled along time axis', size=10, fill=OUTPUT_ACCENT)}"
    )
    a("  </g>")

    # ---- Training data (bottom center) ----
    train_x, train_y = 395, 358
    a(f'  <g transform="translate({train_x}, {train_y})">')
    a(f"    {_rect(0, 0, 200, 95, rx=8, fill=TRAIN_BG, shadow=True)}")
    a(f"    {_text(100, 22, 'Training Data', size=13, fill='#fff', weight='bold')}")
    pair_label = f"{TRAIN_OBS} \u03b8\u2013\u03c8 observations"
    a(f"    {_text(100, 40, pair_label, size=11, fill=TRAIN_TEXT)}")
    a(
        f"    {_text(100, 56, f'{TRAIN_SITES} sites / {TRAIN_PROFILES} profiles', size=11, fill=TRAIN_TEXT)}"
    )
    a(
        f"    {_text(100, 72, f'{TRAIN_SAMPLES} depth-samples', size=11, fill=TRAIN_TEXT)}"
    )
    a(f"    {_text(100, 86, f'{TRAIN_SOURCES} sources', size=10, fill=TRAIN_TEXT)}")
    a("  </g>")

    # Training → RF arrow (upward)
    a(f"  {_line(train_x + 100, train_y - 5, train_x + 100, rf_y + 116)}")

    # ---- Title ----
    a(
        f"  {_text(W // 2, 22, 'Prediction Pipeline', size=11, fill='#888', style='italic')}"
    )

    a("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig 3: pipeline diagram")
    parser.add_argument(
        "--output-dir",
        default="figs/presentation",
        help="Output directory for SVG and PNG",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    svg_text = build_svg()

    svg_path = out / "fig3_pipeline.svg"
    svg_path.write_text(svg_text, encoding="utf-8")
    print(f"Wrote {svg_path}")

    # Render PNG via cairosvg (tspan subscripts degrade to inline text)
    try:
        import cairosvg

        png_path = out / "fig3_pipeline.png"
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(png_path),
            output_width=W * 2,
            output_height=H * 2,
        )
        print(f"Wrote {png_path}")
    except ImportError:
        print(
            "cairosvg not installed; PNG not generated. Install with: uv add cairosvg"
        )


if __name__ == "__main__":
    main()
