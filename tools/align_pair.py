#!/usr/bin/env python3
"""
align_pair.py — prepare an image pair for the before/after wipe slider.

Screen-captured frames often differ by a few pixels in size and position even
when the camera has not moved. In a side-by-side that does not matter; in an
overlaid wipe it reads as a visible jump at the seam and undermines the
comparison. This script finds the sub-pixel-free integer offset between two
frames, corrects it, and crops both to identical dimensions.

It also sanity-checks the pair and refuses to proceed if something is wrong:
  * a frame that is blown out or otherwise carries no recoverable detail
  * a pair whose framing does not actually match (i.e. the camera moved)

Usage:
    python3 tools/align_pair.py OFF.png ON.png OUTPUT_BASENAME

Example:
    python3 tools/align_pair.py \
        raw/plugin_off_overExp_before.png \
        raw/plugin_on_overExp_before.png \
        docs/images/compare_before

    -> writes docs/images/compare_before_off.png
              docs/images/compare_before_on.png

Requires Pillow and NumPy:  pip install pillow numpy
"""

import sys
import os
import numpy as np
from PIL import Image, ImageFilter

# How far to search for the offset, in pixels, in each direction.
SEARCH_RADIUS = 60

# Below this edge-map correlation the two frames are not the same shot.
MIN_CORRELATION = 0.55


def normalise(x):
    """Zero-mean, unit-variance. Removes brightness/contrast differences so the
    comparison responds to structure rather than to the tonal change we expect."""
    return (x - x.mean()) / (x.std() + 1e-6)


def check_has_detail(img, name):
    """Reject frames with no recoverable image data (e.g. clipped to white)."""
    a = np.asarray(img.convert("RGB"))
    interior = a[100:-100, 100:-100] if min(a.shape[:2]) > 240 else a
    uniques = len(np.unique(interior.reshape(-1, 3), axis=0))
    clipped_white = float((a.reshape(-1, 3) == 255).all(1).mean())
    clipped_black = float((a.reshape(-1, 3) == 0).all(1).mean())

    if uniques < 50:
        sys.exit(
            f"ERROR: {name} contains only {uniques} unique colour(s) in its interior.\n"
            f"       There is no recoverable detail in this file. If it came from a\n"
            f"       viewport screengrab of LogC4 data, re-export it from the source\n"
            f"       render or from Resolve with the grade nodes disabled instead."
        )
    if clipped_white > 0.5:
        sys.exit(f"ERROR: {name} is {clipped_white:.1%} pure white. Nothing to align.")
    if clipped_black > 0.5:
        sys.exit(f"ERROR: {name} is {clipped_black:.1%} pure black. Nothing to align.")

    print(f"  {name}: {img.width}x{img.height}, {uniques} unique colours in interior — OK")


def find_offset(a, b):
    """Return (dx, dy, correlation) aligning image a onto image b.

    Correlates edge maps rather than raw pixels, so a large tonal difference
    between the two frames does not swamp the structural match."""
    ea = np.asarray(a.convert("L").filter(ImageFilter.FIND_EDGES), dtype=float)
    eb = np.asarray(b.convert("L").filter(ImageFilter.FIND_EDGES), dtype=float)

    h = min(ea.shape[0], eb.shape[0])
    w = min(ea.shape[1], eb.shape[1])
    A = normalise(ea[:h, :w])
    B = normalise(eb[:h, :w])

    best = (-1.0, 0, 0)
    for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            As = A[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            Bs = B[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            if As.size < 10000:
                continue
            c = float(np.mean(As * Bs))
            if c > best[0]:
                best = (c, dx, dy)

    corr, dx, dy = best
    return dx, dy, corr


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)

    off_path, on_path, out_base = sys.argv[1], sys.argv[2], sys.argv[3]

    for p in (off_path, on_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: no such file: {p}")

    off = Image.open(off_path).convert("RGB")
    on = Image.open(on_path).convert("RGB")

    print("Checking inputs:")
    check_has_detail(off, os.path.basename(off_path))
    check_has_detail(on, os.path.basename(on_path))

    print("\nSearching for offset...")
    dx, dy, corr = find_offset(off, on)
    print(f"  best match: dx={dx}, dy={dy}, edge correlation={corr:.3f}")

    if corr < MIN_CORRELATION:
        sys.exit(
            f"\nERROR: correlation {corr:.3f} is below {MIN_CORRELATION}.\n"
            f"       These two frames do not appear to show the same camera position.\n"
            f"       An overlaid wipe between them would not be a valid comparison.\n"
            f"       Re-capture both frames without moving the camera."
        )

    # Apply the offset and crop both to a common size.
    w = min(off.width, on.width) - abs(dx)
    h = min(off.height, on.height) - abs(dy)
    ox, oy = max(0, dx), max(0, dy)
    nx, ny = max(0, -dx), max(0, -dy)

    off_c = off.crop((ox, oy, ox + w, oy + h))
    on_c = on.crop((nx, ny, nx + w, ny + h))

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    off_out = f"{out_base}_off.png"
    on_out = f"{out_base}_on.png"
    off_c.save(off_out, optimize=True)
    on_c.save(on_out, optimize=True)

    print(f"\nWrote {off_out}  ({w}x{h})")
    print(f"Wrote {on_out}  ({w}x{h})")
    print("\nBoth frames are now identical in size and aligned.")
    print("Un-comment the matching <figure> block in docs/index.html to publish.")


if __name__ == "__main__":
    main()
