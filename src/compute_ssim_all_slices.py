#!/usr/bin/env python3
"""Compute SSIM across all Z-slices of a multi-page TIFF.

Usage examples:
  python compute_ssim_all_slices.py --tiff path/to/stack.tif --mode adjacent --outdir results
  python compute_ssim_all_slices.py --tiff stack.tif --mode pairwise --outdir results

If run without --tiff, the script runs a small self-test on a synthetic stack.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Optional

import numpy as np

try:
    import tifffile
except Exception:
    tifffile = None

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


def load_stack(path: str) -> np.ndarray:
    if tifffile is not None:
        arr = tifffile.imread(path)
    else:
        # fallback to imageio
        import imageio

        arr = imageio.mimread(path)
        arr = np.stack(arr, axis=0)
    # Ensure shape (z, y, x)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr)


def compute_adjacent_ssim(stack: np.ndarray) -> np.ndarray:
    n = stack.shape[0]
    scores = np.zeros(n - 1, dtype=float)
    for i in range(n - 1):
        a = stack[i].astype(np.float32)
        b = stack[i + 1].astype(np.float32)
        dr = max(a.max() - a.min(), b.max() - b.min(), 1.0)
        s = structural_similarity(a, b, data_range=dr)
        scores[i] = s
    return scores


def compute_pairwise_ssim(stack: np.ndarray) -> np.ndarray:
    n = stack.shape[0]
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            a = stack[i].astype(np.float32)
            b = stack[j].astype(np.float32)
            dr = max(a.max() - a.min(), b.max() - b.min(), 1.0)
            s = structural_similarity(a, b, data_range=dr)
            M[i, j] = s
            M[j, i] = s
    return M


def compute_delta_ssim(stack: np.ndarray, delta: int):
    """
    Compute SSIM for sliding pairs (i, i+delta) with step 1.
    Returns a tuple (scores, indices) where scores is an array of length max(0, n - delta)
    and indices contains the starting slice indices i for each comparison (i, i+delta).
    """
    n = stack.shape[0]
    # Support delta=0 meaning adjacent pairs (i, i+1)
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if delta == 0:
        # Equivalent to adjacent
        return compute_adjacent_ssim(stack), np.arange(0, n - 1, dtype=int)
    if n <= delta:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    scores = np.zeros(n - delta, dtype=float)
    indices = np.arange(0, n - delta, dtype=int)
    for idx, i in enumerate(indices):
        a = stack[i].astype(np.float32)
        b = stack[i + delta].astype(np.float32)
        # robust data_range estimation
        dr = max(a.max() - a.min(), b.max() - b.min(), 1.0)
        scores[idx] = structural_similarity(a, b, data_range=dr)
    return scores, indices


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_and_plot_adjacent(scores: np.ndarray, outdir: str, prefix: str = "ssim_adjacent"):
    import matplotlib.pyplot as plt

    np.save(os.path.join(outdir, prefix + ".npy"), scores)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(scores)), scores, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("slice index (i vs i+1)")
    plt.ylabel("SSIM")
    plt.title("SSIM between adjacent Z-slices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, prefix + ".png"))
    plt.close()


def save_and_plot_pairwise(M: np.ndarray, outdir: str, prefix: str = "ssim_pairwise"):
    np.save(os.path.join(outdir, prefix + ".npy"), M)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(M, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("slice index")
    plt.ylabel("slice index")
    plt.title("Pairwise SSIM matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, prefix + ".png"))
    plt.close()


def save_and_plot_delta(scores: np.ndarray, outdir: str, delta: int, prefix: Optional[str] = None):
    if prefix is None:
        prefix = f"ssim_delta_{delta}"
    np.save(os.path.join(outdir, prefix + ".npy"), scores)
    # if indices were passed separately, caller should save them; we'll not assume
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(scores)), scores, marker="o")
    plt.ylim(0, 1)
    if delta == 0:
        xlabel = "slice index (i vs i+1)"
        title = "SSIM for delta=0 (i vs i+1)"
    else:
        xlabel = "slice index (i vs i+delta)"
        title = f"SSIM for delta={delta} (i vs i+{delta})"
    plt.xlabel(xlabel)
    plt.ylabel("SSIM")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, prefix + ".png"))
    plt.close()


def save_and_plot_all_deltas(scores_by_delta: dict[int, np.ndarray], outdir: str,
                              filename: str = "ssim_deltas_summary.png") -> None:
    """Create a single summary figure overlaying SSIM curves for all deltas and the min-SSIM vs delta."""
    # Prepare data
    deltas = sorted(scores_by_delta.keys())
    min_vals = []
    for d in deltas:
        arr = scores_by_delta[d]
        min_vals.append(float(arr.min()) if arr.size else np.nan)
    # Save min curve as .npy
    np.save(os.path.join(outdir, "ssim_min_vs_delta.npy"), np.array([deltas, min_vals], dtype=object))

    plt.figure(figsize=(14, 6))
    # Left subplot: overlay curves per delta
    ax1 = plt.subplot(1, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(deltas))))
    for idx, d in enumerate(deltas):
        y = scores_by_delta[d]
        ax1.plot(np.arange(len(y)), y, label=f"delta={d}", color=colors[idx], linewidth=1.5)
    ax1.set_xlabel("slice index")
    ax1.set_ylabel("SSIM")
    ax1.set_title("SSIM per slice for multiple deltas")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, ncol=2)

    # Right subplot: min-SSIM vs delta
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(deltas, min_vals, marker="o", linewidth=2)
    ax2.set_xlabel("delta")
    ax2.set_ylabel("min SSIM")
    ax2.set_title("Minimum SSIM vs delta")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute SSIM across Z-slices of a TIFF stack")
    p.add_argument("--tiff", type=str, help="Path to multi-page TIFF")
    p.add_argument("--mode", choices=("adjacent", "pairwise", "delta"), default="adjacent")
    p.add_argument("--delta", type=int, action="append",
                   help="Delta value to use when --mode delta. Can be repeated, e.g. --delta 2 --delta 5")
    p.add_argument("--outdir", type=str, default="ssim_results")
    p.add_argument("--start", type=int, default=None, help="Optional start slice index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Optional end slice index (exclusive)")
    p.add_argument("--margin", type=int, default=80,
                   help="If start/end not provided, trim this many slices from start and end (default: 80)")
    args = p.parse_args(argv)

    ensure_outdir(args.outdir)

    if not args.tiff:
        # Run a small self-test
        print("No TIFF provided: running self-test on synthetic stack...")
        z = 10
        base = np.zeros((z, 64, 64), dtype=np.float32)
        for i in range(z):
            rr, cc = np.ogrid[:64, :64]
            base[i] = ((rr - 32) ** 2 + (cc - (16 + i * 3)) ** 2) < (8 + i * 0.5) ** 2
        base = base.astype(np.float32)
        stack = base
    else:
        if not os.path.exists(args.tiff):
            print(f"ERROR: TIFF not found: {args.tiff}")
            return 2
        print(f"Loading TIFF: {args.tiff}")
        stack = load_stack(args.tiff)

    # Keep only requested range. If start/end not provided, optionally trim margins
    if args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end or stack.shape[0]
        stack = stack[s:e]
    else:
        # apply margin trimming if requested and stack is large enough
        margin = int(args.margin or 0)
        if margin > 0 and stack.shape[0] > 2 * margin:
            s = margin
            e = stack.shape[0] - margin
            print(f"Applying margin trim: using slices [{s}:{e}] (margin={margin})")
            stack = stack[s:e]

    n = stack.shape[0]
    print(f"Stack shape: {stack.shape}")

    if args.mode == "adjacent":
        if n < 2:
            print("Not enough slices for adjacent comparison")
            return 1
        scores = compute_adjacent_ssim(stack)
        save_and_plot_adjacent(scores, args.outdir)
        print(f"Saved adjacent SSIM (.npy + .png) in {args.outdir}")
    elif args.mode == "delta":
        # delta mode: may accept multiple --delta values
        deltas = args.delta or [0, 1, 2, 3, 4, 5, 10]
        scores_by_delta: dict[int, np.ndarray] = {}
        for delta in deltas:
            if delta < 0:
                print(f"Skipping invalid delta={delta}")
                continue
            if n <= delta:
                print(f"Skipping delta={delta}: stack has {n} slices (needs > delta)")
                continue
            scores, indices = compute_delta_ssim(stack, delta)
            prefix = f"ssim_delta_{delta}"
            # save both scores and indices
            np.save(os.path.join(args.outdir, prefix + ".npy"), scores)
            np.save(os.path.join(args.outdir, prefix + "_indices.npy"), indices)
            # also save plot
            save_and_plot_delta(scores, args.outdir, delta, prefix=prefix)
            scores_by_delta[delta] = scores
            # print brief stats
            if scores.size > 0:
                print(f"delta={delta}: n_comparisons={scores.size}, mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
            print(f"Saved delta={delta} SSIM (.npy, _indices.npy + .png) in {args.outdir}")

        # After individual deltas, save a single summary figure and min curve
        if scores_by_delta:
            save_and_plot_all_deltas(scores_by_delta, args.outdir, filename="ssim_deltas_summary.png")
            print(f"Saved summary figure (all deltas) and min-SSIM curve in {args.outdir}")
    else:
        # pairwise
        if n > 600:
            warnings.warn(
                f"Pairwise mode will be O(n^2). n={n} may be large and slow; proceed with caution.")
        M = compute_pairwise_ssim(stack)
        save_and_plot_pairwise(M, args.outdir)
        print(f"Saved pairwise SSIM (.npy + .png) in {args.outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
