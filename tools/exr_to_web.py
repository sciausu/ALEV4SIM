#!/usr/bin/env python3
"""
exr_to_web.py — turn a Movie Render Queue EXR into a web-ready PNG.

Going straight from the EXR avoids the two problems that screen captures
introduce: they clip anything outside the display range (which is how the
plugin-on frame ended up as a solid white PNG), and they arrive a few pixels
misaligned, which an overlaid wipe slider exposes as a jump at the seam.

The script:
  * detects and removes the pillarbox padding (the render is 4:3 inside a
    16:9 frame, so ~480 columns are black bars)
  * applies the requested transform
  * reports the value range so you can see whether the data actually fits
    the transform you asked for

Usage:
    python3 tools/exr_to_web.py INPUT.exr OUTPUT.png [--mode MODE] [--exposure STOPS]

Modes:
    linear      treat as scene-linear, apply an sRGB display transform.
                Correct for a standard Unreal linear render.
    normalize   divide by the maximum value, then sRGB. Use for data that
                does not fit 0-1 and whose encoding is unknown.
    logc4       treat as ARRI LogC4, decode to linear, then sRGB.
    raw         clip to 0-1 and apply sRGB, no scaling. Shows you exactly
                what a naive viewer does with the file.

Examples:
    python3 tools/exr_to_web.py plugin-off_overExp.exr docs/images/off.png --mode linear
    python3 tools/exr_to_web.py plugin-on_overExp.exr  docs/images/on.png  --mode normalize

Requires:  pip install OpenEXR numpy pillow
"""

import sys
import argparse
import numpy as np
from PIL import Image

try:
    import OpenEXR
except ImportError:
    sys.exit("ERROR: pip install OpenEXR")

# ---- ARRI LogC4 constants (ARRI white paper, 2022) ----
A = (2 ** 18 - 16) / 117.45
B = (1023 - 95) / 1023
C = 95 / 1023
S = (7 * np.log(2) * 2 ** -12) / (A * B)
T = (2 ** -12 - 64) / A


def logc4_to_linear(y):
    y = np.asarray(y, np.float64)
    return np.where(y < 0, y * S + T, (2 ** (14 * (y - C) / B + 6) - 64) / A)


def srgb(x):
    """Linear -> sRGB display encoding."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def load_exr(path):
    f = OpenEXR.File(path)
    part = f.parts[0]
    key = "RGBA" if "RGBA" in part.channels else list(part.channels)[0]
    return np.asarray(part.channels[key].pixels).astype(np.float32)


def crop_pillarbox(a):
    """Trim black padding using the alpha channel if present, else luminance."""
    if a.shape[2] >= 4:
        cov = a[..., 3]
    else:
        cov = a[..., :3].max(2)
    cols = np.where(cov.max(0) > 0)[0]
    rows = np.where(cov.max(1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        return a[..., :3], (0, a.shape[1], 0, a.shape[0])
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    return a[y0:y1, x0:x1, :3], (x0, x1, y0, y1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--mode", default="linear",
                   choices=["linear", "normalize", "logc4", "raw"])
    p.add_argument("--exposure", type=float, default=0.0,
                   help="exposure adjustment in stops, applied in linear light")
    p.add_argument("--no-crop", action="store_true",
                   help="keep the pillarbox padding")
    args = p.parse_args()

    a = load_exr(args.input)
    full = a.shape
    if args.no_crop:
        rgb = a[..., :3]
        box = (0, full[1], 0, full[0])
    else:
        rgb, box = crop_pillarbox(a)

    print(f"{args.input}")
    print(f"  source frame : {full[1]}x{full[0]}")
    print(f"  active area  : {rgb.shape[1]}x{rgb.shape[0]}  "
          f"(cols {box[0]}-{box[1]}, rows {box[2]}-{box[3]})")
    print(f"  value range  : min={rgb.min():.4f}  max={rgb.max():.4f}  "
          f"mean={rgb.mean():.4f}")
    above = float((rgb > 1.0).mean())
    print(f"  above 1.0    : {100 * above:.2f}%")
    if above > 0.25 and args.mode in ("linear", "raw"):
        print("  NOTE: a large share of this image sits above 1.0. In 'linear' or")
        print("        'raw' mode those pixels clip to white. If the result looks")
        print("        blown out, the data is probably not scene-linear — try")
        print("        --mode normalize or --mode logc4.")

    if args.mode == "linear":
        lin = rgb
    elif args.mode == "normalize":
        mx = float(rgb.max())
        print(f"  normalizing by {mx:.4f}")
        lin = rgb / mx if mx > 0 else rgb
    elif args.mode == "logc4":
        lin = logc4_to_linear(rgb)
    else:  # raw
        lin = rgb

    if args.exposure:
        lin = lin * (2.0 ** args.exposure)
        print(f"  exposure     : {args.exposure:+.2f} stops")

    out = (np.clip(srgb(lin), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(args.output, optimize=True)
    print(f"  wrote {args.output}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
