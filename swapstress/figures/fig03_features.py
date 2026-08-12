"""Descriptor Fig 3: the model's inputs, and what they contribute.

Panel a draws the feature stack as stacked CONUS layers -- the dynamic SMAP L3
theta field on top, then one representative 9 km raster per static covariate
group, each labeled with the group's band count and its share of static
permutation importance. Panel b gives the per-feature permutation importance
of the *released* QRF (``swapstress.model.qrf_permutation``): the drop in
median-prediction R2 on the spatial holdout when one feature is shuffled.

Theta dominates by an order of magnitude, so its bar is broken at the static
axis limit and labeled with its value; drawing it to scale would flatten every
other bar into invisibility. Bars are colored by input kind with the validated
categorical trio -- dynamic theta, static landscape covariates, per-sample
descriptors (depth, Rosetta level). Panel a gives every layer its own ramp --
the panel is a schematic with no colorbars, so distinct ramps read as distinct
variables -- keeping theta on the purples ramp to match its bar color.

The representative band shown for each group is that group's top-ranked
feature in the importance table (falling back down the ranking to one that
exists as a band in the group's raster). The theta layer is a single real
retrieval day, gaps and all, because the daily field *is* the input.

Usage:
    uv run python -m swapstress.figures.fig03_features [--importance-csv ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from swapstress.figures import style

IMPORTANCE_CSV = Path(
    "/nas/soils/swapstress/releases/v03_20260729/evaluation/feature_importance"
    "/permutation_importance.csv"
)
FEATURES_DIR = Path("/nas/soils/swapstress/inference/conus_features")
# One real retrieval day, in Fig 7's July 2023 window.
SMAP_THETA_TIF = Path("/nas/soils/smap/SPL3SMP_E/daily_tif/smap_sm_20230707.tif")

OUT_DIR = Path("figs/descriptor")
STEM = "fig03_features"

FIG_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIG_HEIGHT_MM = 120.0

# Static covariate groups: display name, the raster carrying their bands (the
# global ET0 climatology is exported inside the WorldClim stack), and the ramp
# the layer renders in. One ramp per layer, each thematically its own: the
# panel is a schematic with no colorbars, so the distinct ramps say "distinct
# variables" rather than encoding a shared scale. The theta layer keeps
# ``style.SEQUENTIAL_ALT`` so purple stays the dynamic input's color.
GROUPS = {
    "worldclim": ("WorldClim climate", FEATURES_DIR / "worldclim_9km.tif", "viridis"),
    # YlOrBr truncated off its white end so low values stay a visible cream
    # and the layer's coastline does not dissolve into the page.
    "soilgrids": (
        "SoilGrids soil properties",
        FEATURES_DIR / "soilgrids_9km.tif",
        mpl.colors.LinearSegmentedColormap.from_list(
            "ylorbr_deep",
            mpl.colormaps["YlOrBr"]([0.12 + 0.88 * i / 255.0 for i in range(256)]),
        ),
    ),
    "fao": ("FAO HWSD soil units", FEATURES_DIR / "fao_hwsd_9km.tif", "tab20"),
    "global_et0": ("Global reference ET", FEATURES_DIR / "worldclim_9km.tif", "magma"),
    "landsat_bands": (
        "Landsat reflectance",
        FEATURES_DIR / "landsat_bands_9km.tif",
        "bone",
    ),
}

# Input kinds share the validated categorical trio across both panels.
COLOR_STATIC = style.CATEGORICAL[0]
COLOR_SAMPLE = style.CATEGORICAL[1]
COLOR_THETA = style.CATEGORICAL[2]

N_BARS = 15

# Stack geometry: each layer is the CONUS raster sheared into a parallelogram,
# stacked bottom-up with a fixed rise. The y-scale stays near 1 so CONUS keeps
# close to its true Albers proportions -- the earlier 0.42 flattening read as
# a smeared projection rather than a tilted card.
LAYER_SKEW_DEG = -24.0
LAYER_YSCALE = 0.88
LAYER_RISE = 0.60
LAYER_ASPECT = 360.0 / 667.0

SEASONS = {"winter": "winter", "spring": "spring", "summer": "summer", "fall": "fall"}
WC_VARS = {
    "prec": "Precipitation",
    "tavg": "Mean temperature",
    "tmin": "Min. temperature",
    "tmax": "Max. temperature",
}
SOILGRIDS_VARS = {
    "bdod": "Bulk density",
    "cec": "CEC",
    "cfvo": "Coarse fragments",
    "clay": "Clay",
    "sand": "Sand",
    "silt": "Silt",
    "nitrogen": "Nitrogen",
    "ocd": "Org. C density",
    "ocs": "Org. C stock",
    "phh2o": "pH",
    "soc": "Soil organic C",
}
FIXED_LABELS = {
    "theta": "θ (SMAP L3 soil moisture)",
    "depth_cm": "Sample depth",
    "rosetta_level": "Rosetta level",
    "WISE30s_ID": "WISE soil map unit",
    "HWSD2_ID": "HWSD soil map unit",
    "WRB4": "WRB soil group",
    "WRB_PHASES": "WRB phases",
    "WRB2_CODE": "WRB2 soil group",
    "FAO90": "FAO90 soil unit",
    "eto_yearly_sd": "Reference ET, yearly s.d.",
}


def pretty_name(feature: str) -> str:
    """A readable label for a raw feature name; unknown names pass through."""
    if feature in FIXED_LABELS:
        return FIXED_LABELS[feature]
    parts = feature.split("_")
    if parts[0] == "wc" and len(parts) == 3 and parts[1] in WC_VARS:
        return f"{WC_VARS[parts[1]]}, {SEASONS.get(parts[2], parts[2])}"
    if parts[0] == "eto" and len(parts) == 2:
        return f"Reference ET, {SEASONS.get(parts[1], parts[1])}"
    if len(parts) == 3 and parts[0] in SOILGRIDS_VARS and parts[2] == "mean":
        depth = parts[1].replace("cm", " cm").replace("-", "–")
        return f"{SOILGRIDS_VARS[parts[0]]}, {depth}"
    if parts[0].startswith("B") and len(parts) == 3 and parts[2] == "gs":
        stat = "s.d." if parts[1] == "stdDev" else parts[1]
        return f"Landsat {parts[0]} {stat}, grow. season"
    return feature


def bar_color(group: str) -> str:
    if group == "theta":
        return COLOR_THETA
    if group == "depth":
        return COLOR_SAMPLE
    return COLOR_STATIC


def load_importance(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def representative_band(group: str, df: pd.DataFrame) -> str:
    """The group's top-ranked feature that exists as a band in its raster."""
    with rasterio.open(GROUPS[group][1]) as src:
        bands = set(src.descriptions)
    ranked = df.loc[df["group"] == group, "feature"]
    for name in ranked:
        if name in bands:
            return name
    raise ValueError(f"no {group} feature from the importance table is a band")


def read_band(path: Path, band_name: str) -> np.ndarray:
    with rasterio.open(path) as src:
        idx = src.descriptions.index(band_name) + 1
        arr = src.read(idx).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    # The landsat stack carries -9999 fill without a nodata tag; left in, the
    # percentile stretch collapses onto the fill and the layer renders binary.
    arr[arr == -9999.0] = np.nan
    return arr


def read_theta_on_stack_grid() -> np.ndarray:
    """The daily SMAP field warped onto the 9 km static-stack grid."""
    ref_path = GROUPS["soilgrids"][1]
    with rasterio.open(ref_path) as ref:
        dst = np.full(ref.shape, np.nan)
        with rasterio.open(SMAP_THETA_TIF) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return dst


def normalize(arr: np.ndarray) -> np.ndarray:
    """Robust 2-98 percentile stretch to [0, 1] for schematic rendering."""
    lo, hi = np.nanpercentile(arr, [2, 98])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def layer_transform(ax, index: int) -> mtransforms.Transform:
    return (
        mtransforms.Affine2D()
        .scale(1.0, LAYER_YSCALE)
        .skew_deg(LAYER_SKEW_DEG, 0.0)
        .translate(0.0, index * LAYER_RISE)
        + ax.transData
    )


def draw_stack(ax, df: pd.DataFrame) -> None:
    """Stacked CONUS layers: static groups bottom-up, the theta day on top."""
    # Restrict to the released model's landscape groups: an importance table
    # from a wider feature set (the archived pre-release run used for layout
    # checks) may carry groups the pruned model no longer has.
    landscape = df[df["group"].isin(GROUPS)]
    # Negative permutation importance is sampling noise around zero; clipping
    # before summing keeps the shares a partition of the positive signal.
    shares = landscape.groupby("group")["importance_mean"].apply(
        lambda s: s.clip(lower=0).sum()
    )
    shares = (shares / shares.sum()).sort_values()  # ascending: bottom layer first
    counts = landscape["group"].value_counts()

    layers = []
    for group in shares.index:
        band = representative_band(group, df)
        img = read_band(GROUPS[group][1], band)
        if group == "fao":
            # Map-unit IDs are nominal: stretched raw they draw a north-south
            # gradient that implies a value field. A fixed integer hash spreads
            # adjacent units across the ramp so they read as categorical.
            img = np.where(np.isfinite(img), (img * 2654435761) % 97, np.nan)
        img = normalize(img)
        line2 = f"{counts[group]} bands · {shares[group]:.0%} of static importance"
        layers.append((img, GROUPS[group][2], GROUPS[group][0], line2))
    theta_img = normalize(read_theta_on_stack_grid())
    layers.append(
        (
            theta_img,
            style.SEQUENTIAL_ALT,
            "SMAP L3 θ",
            "one day of retrievals · gray = no overpass",
        )
    )

    # Land mask from a static band: under the theta layer it renders the
    # set-wide convention -- gray is land the day's swaths did not cover.
    land = np.isfinite(read_band(GROUPS["soilgrids"][1], "silt_5-15cm_mean"))

    for i, (img, cmap_name, title, subtitle) in enumerate(layers):
        if title.startswith("SMAP"):
            gray_cmap = mpl.colors.ListedColormap([style.NO_DATA_GRAY])
            gray_cmap.set_bad(alpha=0.0)
            ax.imshow(
                np.where(land, 0.0, np.nan),
                cmap=gray_cmap,
                origin="upper",
                extent=(0.0, 1.0, 0.0, LAYER_ASPECT),
                transform=layer_transform(ax, i),
                interpolation="nearest",
                rasterized=True,
                zorder=i + 0.5,
            )
        cmap = plt.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name
        cmap = cmap.copy()
        cmap.set_bad(alpha=0.0)
        ax.imshow(
            img,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            extent=(0.0, 1.0, 0.0, LAYER_ASPECT),
            transform=layer_transform(ax, i),
            interpolation="nearest",
            rasterized=True,
            zorder=i + 1,
        )
        y_mid = i * LAYER_RISE + 0.5 * LAYER_ASPECT * LAYER_YSCALE
        color = COLOR_THETA if title.startswith("SMAP") else style.AXIS_COLOR
        ax.text(
            1.06,
            y_mid + 0.035,
            title,
            fontsize=style.MAX_TEXT_PT,
            color=color,
            va="bottom",
            ha="left",
        )
        ax.text(
            1.06,
            y_mid + 0.02,
            subtitle,
            fontsize=style.MAX_TEXT_PT - 1.5,
            color=style.MUTED_INK,
            va="top",
            ha="left",
        )

    ax.text(
        0.0,
        -0.16,
        "plus per-sample depth and Rosetta level",
        fontsize=style.MAX_TEXT_PT - 1,
        color=style.MUTED_INK,
        va="top",
        ha="left",
        transform=ax.transData,
    )

    skew_dx = abs(np.tan(np.radians(LAYER_SKEW_DEG))) * LAYER_ASPECT * LAYER_YSCALE
    ax.set_xlim(-skew_dx - 0.03, 2.05)
    ax.set_ylim(-0.24, (len(layers) - 1) * LAYER_RISE + LAYER_ASPECT * LAYER_YSCALE)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_importance(ax, df: pd.DataFrame) -> None:
    top = df.head(N_BARS)
    static_max = top.loc[top["group"] != "theta", "importance_mean"].max()
    xlim = static_max * 1.25

    y = np.arange(len(top))[::-1]
    for yi, (_, row) in zip(y, top.iterrows()):
        val = row["importance_mean"]
        clipped = val > xlim
        ax.barh(
            yi,
            min(val, xlim),
            height=0.62,
            color=bar_color(row["group"]),
            zorder=2,
        )
        if clipped:
            # Broken-bar convention: the theta bar runs off the static scale.
            for dx in (0.955, 0.975):
                ax.plot(
                    [xlim * dx, xlim * (dx - 0.012)],
                    [yi - 0.42, yi + 0.42],
                    color="white",
                    linewidth=1.2,
                    zorder=3,
                    clip_on=False,
                )
            ax.text(
                xlim * 0.93,
                yi,
                f"{val:.2f}",
                va="center",
                ha="right",
                fontsize=style.MAX_TEXT_PT - 1,
                color="white",
                zorder=4,
            )
        else:
            ax.errorbar(
                min(val, xlim),
                yi,
                xerr=row["importance_std"],
                ecolor=style.AXIS_COLOR,
                elinewidth=0.5,
                capsize=0,
                zorder=3,
            )

    ax.set_yticks(y, [pretty_name(f) for f in top["feature"]])
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlim(0, xlim)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Permutation importance (ΔR², median prediction)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=lbl)
        for c, lbl in (
            (COLOR_THETA, "dynamic θ"),
            (COLOR_STATIC, "static covariate"),
            (COLOR_SAMPLE, "sample descriptor"),
        )
    ]
    ax.legend(handles=handles, loc="lower right", handlelength=1.0, borderaxespad=0.2)


def build_figure(output_dir=OUT_DIR, importance_csv=IMPORTANCE_CSV) -> Path:
    style.apply()
    df = load_importance(Path(importance_csv))

    fig = plt.figure(
        figsize=style.figsize(FIG_WIDTH_MM, FIG_HEIGHT_MM), layout="constrained"
    )
    ax_stack, ax_bar = fig.subplots(
        1, 2, width_ratios=(1.15, 1.0), gridspec_kw={"wspace": 0.04}
    )

    draw_stack(ax_stack, df)
    style.panel_label(ax_stack, "a", dx=0.0, dy=0.98)

    draw_importance(ax_bar, df)
    style.panel_label(ax_bar, "b", dx=-0.42)

    out = style.save(fig, Path(output_dir) / STEM)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Descriptor Fig 3: feature stack and released-model importance."
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--importance-csv", default=str(IMPORTANCE_CSV))
    args = parser.parse_args(argv)
    build_figure(args.output_dir, args.importance_csv)


if __name__ == "__main__":
    main()
