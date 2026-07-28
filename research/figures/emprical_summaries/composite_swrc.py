"""
Composite SWRC visualization: Rosetta vs ReESH vs ML VG vs Direct model comparison.

This module generates comparison plots showing soil water retention curves from
multiple sources at the same site:
    - Observed ReESH data points
    - Fitted VG curves (from Bayesian JSON)
    - Rosetta VG parameters (prior)
    - ML-predicted VG parameters (finetuned/pretrained)
    - Direct model predictions (log10_suction vs theta)

Usage:
    python viz/emprical_summaries/composite_swrc.py

    Or programmatically:
    from research.figures.emprical_summaries.composite_swrc import plot_composite_swrc
    plot_composite_swrc(site_id, reesh_json, rosetta_df, ml_pred_df, direct_pred_df, out_path)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from swapstress.sources.depth import depth_to_rosetta_level
from swapstress.swrc import psi_from_theta


def _vg_suction(theta, theta_r, theta_s, alpha, n):
    """Inverse van Genuchten for plotting: theta -> suction (cm).

    Uses a looser Se clip and a 1e-3 cm floor than the PTF baseline, matching
    what these figures were drawn with.
    """
    psi = psi_from_theta(theta, theta_r, theta_s, alpha, n, se_eps=1e-9)
    return np.maximum(psi, 1e-3)


def load_fitted_json(json_path):
    """
    Load fitted VG parameters and observations from a Bayesian/deterministic JSON.

    Returns
    -------
    dict
        Keys are depth_cm (float), values are dicts with 'params' and 'obs' keys.
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    data.pop("metadata", None)

    for depth_str, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "Success":
            continue

        try:
            depth_cm = float(depth_str)
        except (TypeError, ValueError):
            continue

        params = entry.get("parameters", {})
        obs_data = entry.get("data", {})

        try:
            results[depth_cm] = {
                "params": {
                    "theta_r": params["theta_r"]["value"],
                    "theta_s": params["theta_s"]["value"],
                    "alpha": params["alpha"]["value"],
                    "n": params["n"]["value"],
                },
                "obs": {
                    "theta": np.array(obs_data.get("theta", [])),
                    "suction_cm": np.array(obs_data.get("suction_cm", [])),
                },
            }
        except (KeyError, TypeError):
            continue

    return results


def get_rosetta_params_for_level(rosetta_df, level, profile_id=None):
    """
    Extract Rosetta VG parameters for a given level from a DataFrame.

    Parameters
    ----------
    rosetta_df : pd.DataFrame
        DataFrame with columns like US_R3H3_L{level}_VG_{param}.
    level : int
        Rosetta level (1-7).
    profile_id : str, optional
        If provided, filter to this profile/station.

    Returns
    -------
    dict or None
        VG parameters {theta_r, theta_s, alpha, n} or None if not found.
    """
    if rosetta_df is None or rosetta_df.empty:
        return None

    # Filter by profile if specified
    if profile_id is not None:
        for col in ["profile_id", "station", "Index"]:
            if col in rosetta_df.columns:
                mask = rosetta_df[col].astype(str) == str(profile_id)
                if mask.any():
                    rosetta_df = rosetta_df[mask]
                    break

    if rosetta_df.empty:
        return None

    row = rosetta_df.iloc[0]
    prefix = f"US_R3H3_L{level}_VG_"

    try:
        return {
            "theta_r": float(
                row.get(f"{prefix}theta_r", row.get(f"{prefix}thetar", np.nan))
            ),
            "theta_s": float(
                row.get(f"{prefix}theta_s", row.get(f"{prefix}thetas", np.nan))
            ),
            "alpha": 10 ** float(row.get(f"{prefix}log10_alpha", np.nan)),
            "n": 10 ** float(row.get(f"{prefix}log10_n", np.nan)),
        }
    except (TypeError, ValueError):
        return None


def get_ml_params_for_level(ml_df, level, profile_id=None):
    """
    Extract ML-predicted VG parameters for a given level.

    Parameters
    ----------
    ml_df : pd.DataFrame
        DataFrame with predicted VG parameters.
    level : int
        Rosetta level (1-7).
    profile_id : str, optional
        If provided, filter to this profile/station.

    Returns
    -------
    dict or None
        VG parameters {theta_r, theta_s, alpha, n} or None if not found.
    """
    if ml_df is None or ml_df.empty:
        return None

    # Filter by profile if specified
    if profile_id is not None:
        for col in ["profile_id", "station", "sample_id"]:
            if col in ml_df.columns:
                mask = ml_df[col].astype(str).str.contains(str(profile_id), case=False)
                if mask.any():
                    ml_df = ml_df[mask]
                    break

    if ml_df.empty:
        return None

    row = ml_df.iloc[0]
    prefix = f"US_R3H3_L{level}_VG_"

    # Try different column naming conventions
    try:
        theta_r = float(row.get(f"{prefix}theta_r", row.get("theta_r", np.nan)))
        theta_s = float(row.get(f"{prefix}theta_s", row.get("theta_s", np.nan)))

        # Alpha and n may be in log10 or natural scale
        if f"{prefix}log10_alpha" in row.index:
            alpha = 10 ** float(row[f"{prefix}log10_alpha"])
        elif f"{prefix}alpha" in row.index:
            alpha = float(row[f"{prefix}alpha"])
        elif "alpha" in row.index:
            alpha = float(row["alpha"])
        else:
            alpha = np.nan

        if f"{prefix}log10_n" in row.index:
            n = 10 ** float(row[f"{prefix}log10_n"])
        elif f"{prefix}n" in row.index:
            n = float(row[f"{prefix}n"])
        elif "n" in row.index:
            n = float(row["n"])
        else:
            n = np.nan

        return {"theta_r": theta_r, "theta_s": theta_s, "alpha": alpha, "n": n}
    except (TypeError, ValueError):
        return None


def get_direct_predictions(direct_df, profile_id, depth_cm):
    """
    Get direct model predictions (theta, suction_cm) for a profile and depth.

    Supports output from compare_approaches.py which has columns:
    - theta, log10_suction_pred_direct, suction_cm_pred_direct
    Or standard format:
    - theta, log10_suction_cm

    Parameters
    ----------
    direct_df : pd.DataFrame
        DataFrame with theta and prediction columns.
    profile_id : str
        Profile identifier.
    depth_cm : float
        Depth in cm.

    Returns
    -------
    tuple (theta_array, suction_cm_array) or (None, None)
    """
    if direct_df is None or direct_df.empty:
        return None, None

    # Filter by profile
    for col in ["profile_id", "station", "sample_id"]:
        if col in direct_df.columns:
            mask = direct_df[col].astype(str).str.contains(str(profile_id), case=False)
            if mask.any():
                direct_df = direct_df[mask]
                break

    if direct_df.empty:
        return None, None

    # Filter by depth (with tolerance)
    if "depth_cm" in direct_df.columns:
        depth_mask = (direct_df["depth_cm"] - depth_cm).abs() <= 5
        direct_df = direct_df[depth_mask]

    if direct_df.empty or "theta" not in direct_df.columns:
        return None, None

    theta = direct_df["theta"].values

    # Try different column names for predictions
    if "suction_cm_pred_direct" in direct_df.columns:
        suction_cm = direct_df["suction_cm_pred_direct"].values
    elif "log10_suction_pred_direct" in direct_df.columns:
        suction_cm = 10 ** direct_df["log10_suction_pred_direct"].values
    elif "log10_suction_cm" in direct_df.columns:
        suction_cm = 10 ** direct_df["log10_suction_cm"].values
    else:
        return None, None

    return theta, suction_cm


def get_vg_predictions(vg_pred_df, profile_id, depth_cm):
    """
    Get VG model predictions (theta, suction_cm) for a profile and depth.

    Supports output from compare_approaches.py which has columns:
    - theta, log10_suction_pred_vg, suction_cm_pred_vg

    Parameters
    ----------
    vg_pred_df : pd.DataFrame
        DataFrame with theta and VG prediction columns.
    profile_id : str
        Profile identifier.
    depth_cm : float
        Depth in cm.

    Returns
    -------
    tuple (theta_array, suction_cm_array) or (None, None)
    """
    if vg_pred_df is None or vg_pred_df.empty:
        return None, None

    # Filter by profile
    for col in ["profile_id", "station", "sample_id"]:
        if col in vg_pred_df.columns:
            mask = vg_pred_df[col].astype(str).str.contains(str(profile_id), case=False)
            if mask.any():
                vg_pred_df = vg_pred_df[mask]
                break

    if vg_pred_df.empty:
        return None, None

    # Filter by depth (with tolerance)
    if "depth_cm" in vg_pred_df.columns:
        depth_mask = (vg_pred_df["depth_cm"] - depth_cm).abs() <= 5
        vg_pred_df = vg_pred_df[depth_mask]

    if vg_pred_df.empty or "theta" not in vg_pred_df.columns:
        return None, None

    theta = vg_pred_df["theta"].values

    # Try different column names for predictions
    if "suction_cm_pred_vg" in vg_pred_df.columns:
        suction_cm = vg_pred_df["suction_cm_pred_vg"].values
    elif "log10_suction_pred_vg" in vg_pred_df.columns:
        suction_cm = 10 ** vg_pred_df["log10_suction_pred_vg"].values
    else:
        return None, None

    return theta, suction_cm


def plot_composite_swrc(
    site_id,
    fitted_json_path=None,
    rosetta_df=None,
    ml_pred_df=None,
    direct_pred_df=None,
    vg_pred_df=None,
    depths=None,
    save_path=None,
    show=False,
    title=None,
):
    """
    Plot composite SWRC comparing multiple sources for a single site.

    Parameters
    ----------
    site_id : str
        Site identifier.
    fitted_json_path : str, optional
        Path to fitted VG JSON (Bayesian or deterministic).
    rosetta_df : pd.DataFrame, optional
        DataFrame with Rosetta VG parameters.
    ml_pred_df : pd.DataFrame, optional
        DataFrame with ML-predicted VG parameters.
    direct_pred_df : pd.DataFrame, optional
        DataFrame with direct model predictions from compare_approaches.py.
    vg_pred_df : pd.DataFrame, optional
        DataFrame with VG model predictions from compare_approaches.py.
    depths : list, optional
        List of depths to plot. If None, uses depths from fitted JSON.
    save_path : str, optional
        Path to save PNG.
    show : bool
        Whether to display the plot.
    title : str, optional
        Custom title.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Load fitted data
    fitted_data = {}
    if fitted_json_path:
        fitted_data = load_fitted_json(fitted_json_path)

    # Determine depths to plot
    if depths is None:
        depths = sorted(fitted_data.keys()) if fitted_data else []
    if not depths:
        print(f"  Warning: No depths to plot for {site_id}")
        return None

    # Setup figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Color map for depths
    n_depths = len(depths)
    colors = plt.cm.plasma(np.linspace(0, 0.85, max(n_depths, 1)))

    # Theta grid for curve plotting
    theta_grid = np.linspace(0.02, 0.60, 200)

    # Track sources present for legend
    sources_present = {
        "obs": False,
        "fitted": False,
        "rosetta": False,
        "ml": False,
        "direct": False,
        "vg_pred": False,
    }

    for idx, depth_cm in enumerate(depths):
        color = colors[idx]
        level = depth_to_rosetta_level(depth_cm)
        depth_label = f"{int(depth_cm)} cm (L{level})"

        # 1) Plot observed ReESH points
        if depth_cm in fitted_data:
            obs = fitted_data[depth_cm].get("obs", {})
            theta_obs = obs.get("theta", np.array([]))
            suction_obs = obs.get("suction_cm", np.array([]))
            if len(theta_obs) > 0 and len(suction_obs) > 0:
                ax.scatter(
                    theta_obs,
                    suction_obs,
                    c=[color],
                    s=30,
                    alpha=0.7,
                    marker="o",
                    edgecolors="white",
                    linewidth=0.5,
                    label=f"Obs {depth_label}" if idx == 0 else None,
                    zorder=5,
                )
                sources_present["obs"] = True

            # 2) Plot fitted VG curve (dashed)
            params = fitted_data[depth_cm].get("params", {})
            if all(k in params for k in ["theta_r", "theta_s", "alpha", "n"]):
                suction_fit = _vg_suction(theta_grid, **params)
                valid = np.isfinite(suction_fit) & (suction_fit > 0)
                ax.plot(
                    theta_grid[valid],
                    suction_fit[valid],
                    "--",
                    color=color,
                    linewidth=2,
                    label=f"Fitted {depth_label}" if idx == 0 else None,
                    zorder=4,
                )
                sources_present["fitted"] = True

        # 3) Plot Rosetta VG curve
        rosetta_params = get_rosetta_params_for_level(rosetta_df, level, site_id)
        if rosetta_params and all(np.isfinite(v) for v in rosetta_params.values()):
            suction_ros = _vg_suction(theta_grid, **rosetta_params)
            valid = np.isfinite(suction_ros) & (suction_ros > 0)
            ax.plot(
                theta_grid[valid],
                suction_ros[valid],
                "--",
                color=color,
                linewidth=1.5,
                alpha=0.7,
                zorder=3,
            )
            sources_present["rosetta"] = True

        # 4) Plot ML-predicted VG curve
        ml_params = get_ml_params_for_level(ml_pred_df, level, site_id)
        if ml_params and all(np.isfinite(v) for v in ml_params.values()):
            suction_ml = _vg_suction(theta_grid, **ml_params)
            valid = np.isfinite(suction_ml) & (suction_ml > 0)
            ax.plot(
                theta_grid[valid],
                suction_ml[valid],
                "-.",
                color=color,
                linewidth=1.5,
                alpha=0.8,
                zorder=3,
            )
            sources_present["ml"] = True

        # 5) Plot direct model predictions
        theta_direct, suction_direct = get_direct_predictions(
            direct_pred_df, site_id, depth_cm
        )
        if theta_direct is not None and len(theta_direct) > 0:
            # Sort by theta for line plot
            sort_idx = np.argsort(theta_direct)
            ax.plot(
                theta_direct[sort_idx],
                suction_direct[sort_idx],
                ":",
                color=color,
                linewidth=2,
                alpha=0.9,
                zorder=3,
            )
            sources_present["direct"] = True

        # 6) Plot VG model predictions (from compare_approaches.py) - solid line
        theta_vg_pred, suction_vg_pred = get_vg_predictions(
            vg_pred_df, site_id, depth_cm
        )
        if theta_vg_pred is not None and len(theta_vg_pred) > 0:
            sort_idx = np.argsort(theta_vg_pred)
            ax.plot(
                theta_vg_pred[sort_idx],
                suction_vg_pred[sort_idx],
                "-",
                color=color,
                linewidth=2,
                alpha=0.9,
                zorder=3,
            )
            sources_present["vg_pred"] = True

    # Configure axes
    ax.set_yscale("log")
    ax.set_xlabel(r"Volumetric Water Content ($\theta$, $cm^3/cm^3$)", fontsize=12)
    ax.set_ylabel("Soil Water Potential (cm H₂O)", fontsize=12)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(1, 1e7)
    ax.grid(True, which="both", ls="--", c="0.75", alpha=0.5)

    # Build legend
    legend_elements = []
    if sources_present["obs"]:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Observed",
            )
        )
    if sources_present["fitted"]:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=2,
                linestyle="--",
                label="Fitted VG (Bayes)",
            )
        )
    if sources_present["rosetta"]:
        legend_elements.append(
            Line2D(
                [0], [0], color="gray", linewidth=1.5, linestyle="--", label="Rosetta"
            )
        )
    if sources_present["ml"]:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=1.5,
                linestyle="-.",
                label="ML Predicted",
            )
        )
    if sources_present["direct"]:
        legend_elements.append(
            Line2D(
                [0], [0], color="gray", linewidth=2, linestyle=":", label="Direct Model"
            )
        )
    if sources_present["vg_pred"]:
        legend_elements.append(
            Line2D(
                [0], [0], color="gray", linewidth=2, linestyle="-", label="VG Pred (RF)"
            )
        )

    # Add depth color legend
    for idx, depth_cm in enumerate(depths):
        level = depth_to_rosetta_level(depth_cm)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=colors[idx],
                linewidth=3,
                label=f"{int(depth_cm)} cm (L{level})",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
    )

    # Title
    if title is None:
        title = f"Composite SWRC — {site_id}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def find_complete_sites(
    fit_results_dir,
    rosetta_parquet=None,
    ml_predictions_parquet=None,
    direct_predictions_parquet=None,
    fit_method="bayes",
    max_sites=5,
):
    """
    Find sites that have all required data sources for composite plotting.

    Parameters
    ----------
    fit_results_dir : str
        Directory containing fitted JSON files.
    rosetta_parquet : str, optional
        Path to Rosetta parameters parquet.
    ml_predictions_parquet : str, optional
        Path to ML predictions parquet.
    direct_predictions_parquet : str, optional
        Path to direct model predictions parquet.
    fit_method : str
        Subdirectory name for fit method (e.g., 'bayes', 'nelder').
    max_sites : int
        Maximum number of sites to return.

    Returns
    -------
    list of str
        Site IDs with complete data.
    """
    # Get fitted JSON files
    json_dir = os.path.join(fit_results_dir, fit_method)
    if not os.path.isdir(json_dir):
        print(f"Warning: Fit results directory not found: {json_dir}")
        return []

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    fitted_sites = {os.path.splitext(f)[0] for f in json_files}

    # Load other data sources to find overlapping sites
    rosetta_sites = set()
    if rosetta_parquet and os.path.exists(rosetta_parquet):
        try:
            rdf = pd.read_parquet(rosetta_parquet)
            for col in ["station", "profile_id", "Index"]:
                if col in rdf.columns:
                    rosetta_sites = set(rdf[col].astype(str).unique())
                    break
        except Exception as e:
            print(f"Warning: Could not load Rosetta parquet: {e}")

    ml_sites = set()
    if ml_predictions_parquet and os.path.exists(ml_predictions_parquet):
        try:
            mdf = pd.read_parquet(ml_predictions_parquet)
            for col in ["station", "profile_id", "sample_id"]:
                if col in mdf.columns:
                    ml_sites = set(mdf[col].astype(str).unique())
                    break
        except Exception as e:
            print(f"Warning: Could not load ML predictions parquet: {e}")

    direct_sites = set()
    if direct_predictions_parquet and os.path.exists(direct_predictions_parquet):
        try:
            ddf = pd.read_parquet(direct_predictions_parquet)
            for col in ["station", "profile_id", "sample_id"]:
                if col in ddf.columns:
                    direct_sites = set(ddf[col].astype(str).unique())
                    break
        except Exception as e:
            print(f"Warning: Could not load direct predictions parquet: {e}")

    # Find sites with all sources
    complete = fitted_sites.copy()
    if rosetta_sites:
        complete &= rosetta_sites
    if ml_sites:
        complete &= ml_sites
    if direct_sites:
        complete &= direct_sites

    # If no complete overlap, fall back to fitted sites only
    if not complete:
        print("Warning: No sites with all data sources. Using fitted sites only.")
        complete = fitted_sites

    return sorted(complete)[:max_sites]


def batch_plot_composite(
    fit_results_dir,
    out_dir,
    rosetta_parquet=None,
    ml_predictions_parquet=None,
    direct_predictions_parquet=None,
    vg_predictions_parquet=None,
    fit_method="bayes",
    max_sites=5,
    show=False,
):
    """
    Generate composite SWRC plots for multiple sites.

    Parameters
    ----------
    fit_results_dir : str
        Directory containing fitted JSON files.
    out_dir : str
        Output directory for PNG files.
    rosetta_parquet : str, optional
        Path to Rosetta parameters parquet.
    ml_predictions_parquet : str, optional
        Path to ML predictions parquet.
    direct_predictions_parquet : str, optional
        Path to direct model predictions parquet (from compare_approaches.py).
    vg_predictions_parquet : str, optional
        Path to VG model predictions parquet (from compare_approaches.py).
    fit_method : str
        Subdirectory name for fit method.
    max_sites : int
        Maximum number of sites to plot.
    show : bool
        Whether to display plots interactively.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load DataFrames once
    rosetta_df = None
    if rosetta_parquet and os.path.exists(rosetta_parquet):
        try:
            rosetta_df = pd.read_parquet(rosetta_parquet)
        except Exception as e:
            print(f"Warning: Could not load Rosetta parquet: {e}")

    ml_df = None
    if ml_predictions_parquet and os.path.exists(ml_predictions_parquet):
        try:
            ml_df = pd.read_parquet(ml_predictions_parquet)
        except Exception as e:
            print(f"Warning: Could not load ML predictions parquet: {e}")

    direct_df = None
    if direct_predictions_parquet and os.path.exists(direct_predictions_parquet):
        try:
            direct_df = pd.read_parquet(direct_predictions_parquet)
        except Exception as e:
            print(f"Warning: Could not load direct predictions parquet: {e}")

    vg_pred_df = None
    if vg_predictions_parquet and os.path.exists(vg_predictions_parquet):
        try:
            vg_pred_df = pd.read_parquet(vg_predictions_parquet)
        except Exception as e:
            print(f"Warning: Could not load VG predictions parquet: {e}")

    # Find sites
    sites = find_complete_sites(
        fit_results_dir,
        rosetta_parquet,
        ml_predictions_parquet,
        direct_predictions_parquet,
        fit_method,
        max_sites,
    )

    if not sites:
        print("No sites found for composite plotting.")
        return

    print(f"Plotting {len(sites)} sites...")

    for site_id in sites:
        json_path = os.path.join(fit_results_dir, fit_method, f"{site_id}.json")
        if not os.path.exists(json_path):
            # Try with _bayes_results or _fit_results suffix
            for suffix in ["_bayes_results.json", "_fit_results.json", ".json"]:
                candidate = os.path.join(
                    fit_results_dir, fit_method, f"{site_id}{suffix}"
                )
                if os.path.exists(candidate):
                    json_path = candidate
                    break

        out_path = os.path.join(out_dir, f"{site_id}_composite_swrc.png")

        try:
            plot_composite_swrc(
                site_id=site_id,
                fitted_json_path=json_path if os.path.exists(json_path) else None,
                rosetta_df=rosetta_df,
                ml_pred_df=ml_df,
                direct_pred_df=direct_df,
                vg_pred_df=vg_pred_df,
                save_path=out_path,
                show=show,
            )
        except Exception as e:
            print(f"  Error plotting {site_id}: {e}")


def main():
    import argparse

    data_root = os.path.join("/nas", "soils")

    parser = argparse.ArgumentParser(
        description="Generate composite SWRC comparison plots from fitted data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fit-dir",
        "-f",
        default=os.path.join(
            data_root, "soil_potential_obs", "curve_fits", "mt_mesonet"
        ),
        help="Directory containing fitted JSON files (with method subdirs)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default=os.path.join(
            data_root, "soil_potential_obs", "composite_swrc_plots", "mt_mesonet"
        ),
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--method",
        "-m",
        default="bayes",
        choices=["bayes", "nelder", "leastsq", "powell"],
        help="Fitting method subdirectory to use",
    )
    parser.add_argument(
        "--max-sites",
        "-n",
        type=int,
        default=10,
        help="Maximum number of sites to plot",
    )
    parser.add_argument(
        "--rosetta",
        default=None,
        help="Path to Rosetta parameters parquet (optional)",
    )
    parser.add_argument(
        "--ml-pred",
        default=None,
        help="Path to ML predictions parquet (optional)",
    )
    parser.add_argument(
        "--direct-pred",
        default=None,
        help="Path to direct model predictions parquet (from compare_approaches.py)",
    )
    parser.add_argument(
        "--vg-pred",
        default=None,
        help="Path to VG model predictions parquet (from compare_approaches.py)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    batch_plot_composite(
        fit_results_dir=args.fit_dir,
        out_dir=args.out_dir,
        rosetta_parquet=args.rosetta,
        ml_predictions_parquet=args.ml_pred,
        direct_predictions_parquet=args.direct_pred,
        vg_predictions_parquet=args.vg_pred,
        fit_method=args.method,
        max_sites=args.max_sites,
        show=args.show,
    )


if __name__ == "__main__":
    main()
# ========================= EOF ====================================================================
