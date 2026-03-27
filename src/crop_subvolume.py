"""Crop a 3D subvolume from a TIFF image using tifffile.

Example:
    python crop_subvolume.py \
        --input input.tif \
        --output cropped.tif \
        --z 10 50 \
        --y 100 300 \
        --x 200 600

Coordinates follow Python slicing rules: start inclusive, stop exclusive.
"""

import argparse
import sys

import numpy as np
import tifffile


def parse_args():
    """CLI setting."""
    parser = argparse.ArgumentParser(
        description="""
Crop a 3D subvolume from a TIFF image.

The input TIFF must contain at least (Z, Y, X).
Ranges follow Python slicing convention: start is inclusive, stop is exclusive.

Example:
    crop_subvolume.py --input volume.tif --output crop.tif --z 10 60 --y 100 300 --x 50 200

This extracts:
    Z = 10 .. 59
    Y = 100 .. 299
    X = 50 .. 199
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True, help="Path to input TIFF file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save cropped TIFF.")

    parser.add_argument(
        "--z",
        nargs=2,
        type=int,
        metavar=("Z_START", "Z_STOP"),
        required=True,
        help="Z-range to crop (start stop).",
    )
    parser.add_argument(
        "--y",
        nargs=2,
        type=int,
        metavar=("Y_START", "Y_STOP"),
        required=True,
        help="Y-range to crop (start stop).",
    )
    parser.add_argument(
        "--x",
        nargs=2,
        type=int,
        metavar=("X_START", "X_STOP"),
        required=True,
        help="X-range to crop (start stop).",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show an ASCII diagram explaining 3D cropping and exit.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def validate_range(rng, max_size, axis_name):
    """Validate the range given by the user or exit."""
    start, stop = rng
    if start < 0 or stop < 0 or start >= stop:
        sys.exit(f"Invalid {axis_name}-range: {rng}")
    if stop > max_size:
        sys.exit(f"{axis_name}-range {rng} exceeds image size {max_size}")


def main():
    """Take the arguments of the CLI and crop the image."""
    args = parse_args()

    # Load image
    img = tifffile.imread(args.input)

    if img.ndim < 3:
        sys.exit("Input image must be at least 3D (Z,Y,X).")

    # Handle images with channels or extra dims
    # Common shapes: (Z,Y,X), (C,Z,Y,X), (Z,Y,X,C)
    # Adjust slicing accordingly

    # Determine which axes correspond to Z,Y,X
    # Heuristic: last two axes are Y,X; remaining spatial axis is Z
    if img.ndim == 3:
        zdim, ydim, xdim = img.shape
        slicer = (slice(*args.z), slice(*args.y), slice(*args.x))

    elif img.ndim == 4:
        # Case 1: (C,Z,Y,X)
        # Case 2: (Z,Y,X,C)
        if img.shape[-1] not in (args.x[1], args.y[1], args.z[1]):
            # Assume (C,Z,Y,X)
            cdim, zdim, ydim, xdim = img.shape
            validate_range(args.z, zdim, "Z")
            validate_range(args.y, ydim, "Y")
            validate_range(args.x, xdim, "X")
            slicer = (slice(None), slice(*args.z), slice(*args.y), slice(*args.x))
        else:
            # Assume (Z,Y,X,C)
            zdim, ydim, xdim, cdim = img.shape
            validate_range(args.z, zdim, "Z")
            validate_range(args.y, ydim, "Y")
            validate_range(args.x, xdim, "X")
            slicer = (slice(*args.z), slice(*args.y), slice(*args.x), slice(None))

    else:
        sys.exit(f"Unsupported image dimensionality: {img.ndim}D")

    # Validate basic ranges for 3D case
    if img.ndim == 3:
        validate_range(args.z, img.shape[0], "Z")
        validate_range(args.y, img.shape[1], "Y")
        validate_range(args.x, img.shape[2], "X")

    # Crop
    subvol = img[slicer]

    # Save
    tifffile.imwrite(args.output, subvol)
    print(f"Saved cropped subvolume to {args.output}")


if __name__ == "__main__":
    main()
