#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tifffile as tiff


def to_binary(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    Convertit un array quelconque en binaire {0,1}.
    - Si uint8 avec max>1: on suppose 0/255 ou probabilités scalées.
    - Si float: threshold.
    """
    if x.dtype.kind in ("u", "i"):  # int
        # cas typique 0/1 ou 0/255
        if x.max() <= 1:
            return (x > 0).astype(np.uint8)
        return (x > 127).astype(np.uint8)  # 0/255 => >127
    else:
        # float => thr
        return (x > thr).astype(np.uint8)


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8):
    """
    pred, gt: {0,1}
    Retourne dice, iou, precision, recall
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float(dice), float(iou), float(precision), float(recall), float(tp), float(fp), float(fn)


def per_slice_metrics(pred3d, gt3d, axis=0):
    """
    Calcule dice/iou par slice (utile pour voir où ça foire).
    axis=0 correspond à (z,y,x) slice par slice.
    """
    pred3d = np.moveaxis(pred3d, axis, 0)
    gt3d = np.moveaxis(gt3d, axis, 0)
    dices, ious = [], []
    for i in range(pred3d.shape[0]):
        d, j, *_ = dice_iou(pred3d[i], gt3d[i])
        dices.append(d)
        ious.append(j)
    return np.array(dices), np.array(ious)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="TIFF prédiction (2D ou 3D)")
    ap.add_argument("--gt", required=True, help="TIFF vérité terrain (2D ou 3D)")
    ap.add_argument("--thr", type=float, default=0.5, help="Seuil si images float (default 0.5)")
    ap.add_argument("--per_slice", action="store_true", help="Calcule aussi les stats par slice")
    ap.add_argument("--axis", type=int, default=0, help="Axe des slices si 3D (default 0)")
    ap.add_argument("--save_csv", default="", help="Chemin CSV pour sauver les métriques par slice")
    args = ap.parse_args()

    pred = tiff.imread(args.pred)
    gt = tiff.imread(args.gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pred_b = to_binary(pred, thr=args.thr)
    gt_b = to_binary(gt, thr=args.thr)

    dice, iou, precision, recall, tp, fp, fn = dice_iou(pred_b, gt_b)

    print("=== Global metrics ===")
    print(f"pred: {args.pred}")
    print(f"gt  : {args.gt}")
    print(f"shape={pred.shape} dtype_pred={pred.dtype} dtype_gt={gt.dtype}")
    print(f"Dice     : {dice:.6f}")
    print(f"IoU      : {iou:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"TP={int(tp)} FP={int(fp)} FN={int(fn)}")

    # Optionnel: métriques par slice
    if args.per_slice:
        if pred_b.ndim != 3:
            raise ValueError("--per_slice nécessite un volume 3D (D,H,W).")
        dices, ious = per_slice_metrics(pred_b, gt_b, axis=args.axis)

        print("\n=== Per-slice summary ===")
        print(f"axis={args.axis}  n_slices={len(dices)}")
        print(f"Dice mean={dices.mean():.6f}  min={dices.min():.6f}  p05={np.percentile(dices,5):.6f}")
        print(f"IoU  mean={ious.mean():.6f}   min={ious.min():.6f}   p05={np.percentile(ious,5):.6f}")

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
            import csv
            with open(args.save_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["slice_index", "dice", "iou"])
                for i, (d, j) in enumerate(zip(dices, ious)):
                    w.writerow([i, float(d), float(j)])
            print(f"[Saved CSV] {args.save_csv}")


if __name__ == "__main__":
    main()
