"""Animated GIF of 2024 CONUS daily SMAP L3 soil moisture (gap-filled).

Reads from the gap-filled directory (spatially + temporally complete).

Output written to /tmp/conus_smap_2024.gif.

Usage:
    uv run python viz/inferences/animate_conus_smap.py
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

SMAP_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif_gapfilled")
SMAP_RAW_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif")
OUT_GIF = Path("/tmp/conus_smap_2024.gif")

VMIN, VMAX = 0.02, 0.55  # m³/m³ — p2/p98 across 2024
FPS = 12
CMAP = "viridis"

_RE = re.compile(r"^smap_sm_(\d{8})\.tif$")


def _discover(start: date, end: date) -> dict[date, Path]:
    """Gapfilled dir first; fall back to raw for any missing dates."""
    result: dict[date, Path] = {}
    for directory in [SMAP_DIR, SMAP_RAW_DIR]:
        for path in sorted(directory.glob("smap_sm_*.tif")):
            m = _RE.match(path.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            if start <= d <= end and d not in result:
                result[d] = path
    return dict(sorted(result.items()))


def _load(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data


def _render_frame(data: np.ndarray, d: date) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=80)
    ax.imshow(
        data,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_axis_off()
    ax.set_title(d.strftime("%Y-%m-%d"), fontsize=11, pad=4)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("soil moisture (m³/m³)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main() -> None:
    start, end = date(2024, 1, 1), date(2024, 12, 31)
    rasters = _discover(start, end)

    print(f"SMAP frames found: {len(rasters)}")

    frames: list[Image.Image] = []
    for i, (d, path) in enumerate(rasters.items()):
        img = _render_frame(_load(path), d)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256))
        if (i + 1) % 30 == 0 or (i + 1) == len(rasters):
            print(f"  rendered {i + 1}/{len(rasters)}")

    duration_ms = int(1000 / FPS)
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    size_mb = OUT_GIF.stat().st_size / 1e6
    print(f"Wrote {OUT_GIF}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
