"""Convert .mp4 files in a directory to .gif, placing both formats in a subdirectory.

Usage:
    python -m viz.mp4_to_gif /path/to/mp4s --outdir ts_anims --scale 0.7 --duration 200
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


def convert_mp4_to_gif(src_dir, outdir="ts_anims", scale=0.7, duration=200, max_mb=50):
    src = Path(src_dir)
    dst = src / outdir
    dst.mkdir(exist_ok=True)

    import shutil

    for mp4 in sorted(src.glob("*.mp4")):
        print(f"Converting {mp4.name} ...")
        frames = iio.imread(mp4, plugin="pyav")
        frames = frames[::2]

        h, w = frames.shape[1], frames.shape[2]
        new_h, new_w = int(h * scale), int(w * scale)

        resized = []
        for f in frames:
            img = Image.fromarray(f).resize((new_w, new_h), Image.LANCZOS)
            resized.append(np.array(img))
        resized = np.stack(resized)

        gif_path = dst / mp4.with_suffix(".gif").name
        iio.imwrite(gif_path, resized, plugin="pillow", duration=duration, loop=0)

        shutil.copy2(mp4, dst / mp4.name)

        size_mb = gif_path.stat().st_size / (1024 * 1024)
        flag = " *** OVER LIMIT ***" if size_mb > max_mb else ""
        print(f"  -> {new_h}x{new_w}, {size_mb:.1f} MB{flag}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mp4 animations to gif")
    parser.add_argument("src_dir", help="Directory containing .mp4 files")
    parser.add_argument(
        "--outdir",
        default="ts_anims",
        help="Subdirectory name for output (default: ts_anims)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.7,
        help="Resolution scale factor (default: 0.7)",
    )
    parser.add_argument(
        "--duration", type=int, default=200, help="Frame duration in ms (default: 200)"
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=50,
        help="Warn if gif exceeds this size in MB (default: 50)",
    )
    args = parser.parse_args()

    convert_mp4_to_gif(
        args.src_dir, args.outdir, args.scale, args.duration, args.max_mb
    )
