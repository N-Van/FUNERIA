# src/utils/redim_urn_manualcrop.py
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import zoom
from tqdm import tqdm


def clamp_range(a: int, b: int, n: int, name: str):
    a2 = max(0, min(n, a))
    b2 = max(0, min(n, b))
    if a2 >= b2:
        raise ValueError(f"Invalid {name} range after clamping: [{a},{b}] -> [{a2},{b2}] with size={n}")
    return a2, b2


def zoom_isotropic_with_progress(vol: np.ndarray, factor: float, order: int = 1) -> np.ndarray:
    """Isotropic zoom """
    if factor <= 0:
        raise ValueError("factor must be > 0")

    Z, Y, X = vol.shape

    # zoom Z
    newZ = max(1, int(round(Z * factor)))
    tqdm.write(f"[zoom] Step 1/2: Z {Z} -> {newZ} (factor {factor:.6f})")
    vol_z = zoom(vol, (factor, 1.0, 1.0), order=order)

    # zoom YX per slice
    Z2, Y2, X2 = vol_z.shape
    newY = max(1, int(round(Y2 * factor)))
    newX = max(1, int(round(X2 * factor)))
    tqdm.write(f"[zoom] Step 2/2: YX {Y2}x{X2} -> {newY}x{newX} (factor {factor:.6f})")

    out = np.empty((Z2, newY, newX), dtype=vol.dtype)
    for z in tqdm(range(Z2), desc="Resizing slices (YX)", unit="slice"):
        out[z] = zoom(vol_z[z], (factor, factor), order=order)

    return out


def pad_to_cube_center(vol: np.ndarray, pad_value: float = 0.0):
    """Pad (centered) to make the volume cubic with side = max(Z,Y,X)"""
    Z, Y, X = vol.shape
    S = max(Z, Y, X)

    def split_pad(cur, target):
        total = target - cur
        before = total // 2
        after = total - before
        return before, after

    pz0, pz1 = split_pad(Z, S)
    py0, py1 = split_pad(Y, S)
    px0, px1 = split_pad(X, S)

    padded = np.pad(
        vol,
        pad_width=((pz0, pz1), (py0, py1), (px0, px1)),
        mode="constant",
        constant_values=pad_value,
    )
    return padded, (pz0, pz1, py0, py1, px0, px1), S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--out_size", type=int, default=640)
    p.add_argument("--order", type=int, default=1)
    #  manual bbox 
    p.add_argument("--z", nargs=2, type=int, required=True, metavar=("Z0", "Z1"))
    p.add_argument("--y", nargs=2, type=int, required=True, metavar=("Y0", "Y1"))
    p.add_argument("--x", nargs=2, type=int, required=True, metavar=("X0", "X1"))

    p.add_argument("--meta", type=str, default=None, help="Optional path to save transform metadata as JSON.")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    tqdm.write("[1/4] Loading TIFF...")
    vol = tiff.imread(str(inp))
    if vol.ndim != 3:
        raise ValueError(f"Expected (Z,Y,X), got shape {vol.shape}")
    vol = vol.astype(np.float32)
    orig_shape = tuple(vol.shape)
    tqdm.write(f"     Loaded shape={orig_shape} dtype=float32")

    tqdm.write("[2/4] Cropping with provided ranges...")
    Z, Y, X = vol.shape
    z0, z1 = clamp_range(args.z[0], args.z[1], Z, "Z")
    y0, y1 = clamp_range(args.y[0], args.y[1], Y, "Y")
    x0, x1 = clamp_range(args.x[0], args.x[1], X, "X")
    crop = vol[z0:z1, y0:y1, x0:x1]
    tqdm.write(f"     Crop: Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] -> {crop.shape}")

    tqdm.write("[3/4] Padding to cube...")
    cube, pads, cube_size = pad_to_cube_center(crop, pad_value=0.0)
    pz0, pz1, py0, py1, px0, px1 = pads
    tqdm.write(f"Padded to cube: {cube.shape} (side={cube_size})")
    tqdm.write(f" Pads: Z({pz0},{pz1}) Y({py0},{py1}) X({px0},{px1})")

    tqdm.write("[4/4] Isotropic resize to target cube...")
    target = int(args.out_size)
    factor = target / float(cube_size)
    tqdm.write(f"     cube_side={cube_size} -> target={target} => isotropic factor={factor:.6f}")
    out_vol = zoom_isotropic_with_progress(cube, factor=factor, order=args.order)

    if out_vol.shape != (target, target, target):
        fixed = np.zeros((target, target, target), dtype=out_vol.dtype)
        z = min(target, out_vol.shape[0])
        y = min(target, out_vol.shape[1])
        x = min(target, out_vol.shape[2])
        fixed[:z, :y, :x] = out_vol[:z, :y, :x]
        out_vol = fixed
        tqdm.write(f" Note: adjusted output to exact cube {out_vol.shape}")

    tiff.imwrite(str(out), out_vol.astype(np.float32))
    tqdm.write(f" Saved: {out} shape={out_vol.shape}")

    meta = {
        "input": str(inp),
        "output": str(out),
        "orig_shape_ZYX": list(orig_shape),
        "crop_ranges_ZYX": {"z": [z0, z1], "y": [y0, y1], "x": [x0, x1]},
        "crop_shape_ZYX": list(crop.shape),
        "pad_to_cube": {"pads_ZYX": {"z": [pz0, pz1], "y": [py0, py1], "x": [px0, px1]},
                        "cube_side": int(cube_size)},
        "resize": {"out_size": int(target), "isotropic_factor": float(factor), "order": int(args.order)},
        "note_reconstruction": "To reconstruct: inverse zoom by 1/factor -> remove padding -> paste back into original at crop ranges."
    }

    if args.meta is not None:
        meta_path = Path(args.meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        tqdm.write(f" Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
