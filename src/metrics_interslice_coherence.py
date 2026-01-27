# src/metrics_interslice_coherence.py
import os
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff

def load_tiff_any(path: str) -> np.ndarray:
    """
    Robust loading:
    - try memmap for speed
    - fallback to imread if not mappable (common with some TIFF compressions)
    Returns numpy array.
    """
    try:
        arr = tiff.memmap(path)
        # memmap may fail later for some files; force simple access
        _ = arr.shape
        return arr
    except Exception as e:
        arr = tiff.imread(path)
        return arr

def to_binary(vol: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    Convert volume to boolean mask.
    Accepts:
      - uint8 {0,255} or {0,1}
      - float probs [0,1]
      - anything numeric
    thr is used only if values are not obviously binary.
    """
    v = vol
    if v.dtype == np.bool_:
        return v

    # If looks like 0/255
    if np.issubdtype(v.dtype, np.integer):
        vmax = int(v.max()) if v.size else 0
        if vmax > 1:
            # common case: 0/255
            return (v > 0)
        else:
            return (v > 0)

    # float / other: threshold
    return (v > thr)

def move_axis_to_depth(vol: np.ndarray, axis: int) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")
    return np.moveaxis(vol, axis, 0)  # (D,H,W)

def dice_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7):
    """
    a,b: boolean 2D arrays
    """
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    sa = a.sum(dtype=np.int64)
    sb = b.sum(dtype=np.int64)
    union = sa + sb - inter
    dice = (2.0 * inter) / (sa + sb + eps)
    iou  = inter / (union + eps)
    return float(dice), float(iou), int(sa), int(sb), int(inter), int(union)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predicted 3D tiff")
    ap.add_argument("--axis", type=int, default=0, help="0=z, 1=y, 2=x (axis to treat as depth)")
    ap.add_argument("--thr", type=float, default=0.5, help="threshold if pred is float/prob (ignored for uint8 masks)")
    ap.add_argument("--z_start", type=int, default=0, help="start slice index (after axis move)")
    ap.add_argument("--z_len", type=int, default=-1, help="number of slices to use (-1=all possible)")
    ap.add_argument("--save_csv", default="", help="optional csv output")
    args = ap.parse_args()

    pred = load_tiff_any(args.pred)
    pred_dhw = move_axis_to_depth(pred, args.axis)
    D, H, W = pred_dhw.shape

    pred_bin = to_binary(pred_dhw, thr=args.thr)

    z0 = max(0, args.z_start)
    z1 = D if args.z_len < 0 else min(D, z0 + args.z_len)
    if z1 - z0 < 2:
        raise ValueError(f"Need at least 2 slices to compute consecutive metrics. Got range [{z0},{z1})")

    rows = []
    for z in range(z0, z1 - 1):
        a = pred_bin[z]
        b = pred_bin[z + 1]
        d, j, sa, sb, inter, uni = dice_iou(a, b)
        rows.append({
            "z": z,
            "dice_z_z+1": d,
            "iou_z_z+1": j,
            "area_z": sa,
            "area_z+1": sb,
            "inter": inter,
            "union": uni,
            "delta_area": int(sb - sa),
            "abs_delta_area": int(abs(sb - sa)),
        })

    df = pd.DataFrame(rows)

    # summaries
    print("=== Inter-slice coherence (pred vs pred[z+1]) ===")
    print(f"pred: {args.pred}")
    print(f"axis={args.axis}  depth_range=[{z0},{z1})  pairs={len(df)}")
    print(f"Mean Dice: {df['dice_z_z+1'].mean():.6f}")
    print(f"Mean IoU : {df['iou_z_z+1'].mean():.6f}")
    print(f"Median Dice: {df['dice_z_z+1'].median():.6f}")
    print(f"Median IoU : {df['iou_z_z+1'].median():.6f}")
    print(f"P10/P90 Dice: {df['dice_z_z+1'].quantile(0.10):.6f} / {df['dice_z_z+1'].quantile(0.90):.6f}")
    print(f"P10/P90 IoU : {df['iou_z_z+1'].quantile(0.10):.6f} / {df['iou_z_z+1'].quantile(0.90):.6f}")
    print(f"Mean |Δarea|: {df['abs_delta_area'].mean():.2f} px")

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"[Saved] {args.save_csv}")

if __name__ == "__main__":
    main()
