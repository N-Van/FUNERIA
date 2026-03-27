import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
import cv2
import gc
from typing import List, Dict
import argparse
import tifffile

def normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)

def apply_clahe(sl, use_clahe=True):
    sl_prep = sl.copy()
    if not use_clahe:
        return sl_prep
    sl_prep = normalize_to_uint8(sl_prep)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(sl_prep)

def convert_rgb(picture_3D: np.ndarray) -> np.ndarray:
    if picture_3D.ndim == 2:
        return np.repeat(picture_3D[..., np.newaxis], 3, -1)
    return picture_3D

def get_urna_mask_threshold(img_gray, thr=60):
    if img_gray.ndim != 2:
        raise ValueError(f"Attend une image 2D, reçu shape={img_gray.shape}")
    urna_mask = (img_gray >= thr).astype(np.uint8) * 255
    return urna_mask

def make_grid_points(h, w, stride, label=1):
    xs = np.arange(stride // 2, w, stride)
    ys = np.arange(stride // 2, h, stride)
    pts = [(int(x), int(y)) for y in ys for x in xs]
    return pts

def compute_dice_score(pred_masks, gt_mask):
    if not pred_masks or gt_mask.sum() == 0:
        return 0.0
    full_pred = np.logical_or.reduce(pred_masks)
    binary_gt = (gt_mask > 0)
    intersection = np.logical_and(full_pred, binary_gt).sum()
    total = full_pred.sum() + binary_gt.sum()
    return (2.0 * intersection) / total if total > 0 else 1.0

def sam_inference(model, img_rgb, urna_mask=None, mode="grid", grid_stride=64, 
                  points=None, labels=None, device="cuda", imgsz=640, iou=0.25, 
                  max_det=100, conf=0.75, min_area=300, max_area_ratio=0.05):
    H, W, _ = img_rgb.shape
    img_area = H * W
    max_area = int(max_area_ratio * img_area)
    pts = make_grid_points(H, W, grid_stride)
    f_points = [(x, y) for (x, y) in pts if urna_mask[y, x]] if urna_mask is not None else pts
    f_labels = [1] * len(f_points)
    if not f_points:
        return [], {"n_masks": 0}
    with torch.inference_mode():
        results = model(img_rgb, points=f_points, labels=f_labels, device=device, 
                        imgsz=imgsz, iou=iou, max_det=max_det, conf=conf, agnostic_nms=True)
    res = results[0]
    if res.masks is None:
        return [], {"n_masks": 0}
    masks = res.masks.data.cpu().numpy().astype(np.uint8)
    if masks.ndim == 2: masks = masks[None, ...]
    filtered = [m > 0 for m in masks if min_area <= (m > 0).sum() <= max_area]
    return filtered, {"n_masks": len(filtered)}

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class SupervisedTrainer:
    def __init__(self, model, volume, gt_volume, config=None):
        self.model = model
        self.volume = volume
        self.gt_volume = gt_volume
        self.config = config or {}
        self.history = {'slice_idx': [], 'dice': [], 'n_masks': []}

    def eval_step(self, idx):
        clear_gpu_memory()
        img = convert_rgb(apply_clahe(self.volume[idx]))
        urna = get_urna_mask_threshold(self.volume[idx])
        preds, _ = sam_inference(self.model, img, urna, **self.config['sam_params'])
        dice = compute_dice_score(preds, self.gt_volume[idx])
        return {'dice': dice, 'n_masks': len(preds)}

    def optimize(self, n_iter=5):
        best_s, best_p = -1.0, None
        for i in range(n_iter):
            p = {'mode': 'grid', 'grid_stride': int(np.random.choice([64, 80, 100])),
                 'conf': float(np.random.uniform(0.3, 0.6)), 'min_area': 300, 'max_det': 50}
            self.config['sam_params'] = p
            test_idx = np.random.randint(0, len(self.volume))
            res = self.eval_step(test_idx)
            print(f"Essai {i+1}: Dice {res['dice']:.4f}")
            if res['dice'] > best_s:
                best_s, best_p = res['dice'], p
        return best_p, best_s

def plot_clean_results(df, path):
    plt.figure(figsize=(10, 5))
    mean_dice = df["dice"].mean()
    plt.plot(df["slice_idx"], df["dice"], color="#2c3e50", linewidth=2, label="Score de Dice")
    plt.fill_between(df["slice_idx"], df["dice"], color="#3498db", alpha=0.2)
    plt.axhline(y=mean_dice, color="#e74c3c", linestyle="--", label=f"Moyenne: {mean_dice:.2f}")
    plt.ylim(0, 1.05)
    plt.xlabel("Profondeur du volume (Slice)")
    plt.ylabel("Précision (Dice Score)")
    plt.title("Performance de la segmentation SAM vs Vérité Terrain")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--model', default='sam2.1_b.pt')
    parser.add_argument('--output', default='./results')
    parser.add_argument('--n_steps', type=int, default=15)
    args = parser.parse_args()

    vol = tifffile.imread(args.volume)
    gt = tifffile.imread(args.gt)
    
    if vol.ndim == 3 and vol.shape[2] < vol.shape[0]:
        vol = np.transpose(vol, (2, 0, 1))
    if gt.ndim == 3 and gt.shape[2] < gt.shape[0]:
        gt = np.transpose(gt, (2, 0, 1))

    from ultralytics import SAM
    model = SAM(args.model)
    trainer = SupervisedTrainer(model, vol, gt)
    
    print("Phase 1: Optimisation des paramètres")
    best_p, _ = trainer.optimize(n_iter=8)
    trainer.config['sam_params'] = best_p

    print("Phase 2: Validation complète")
    slices = np.linspace(0, len(vol)-1, args.n_steps, dtype=int)
    for idx in slices:
        res = trainer.eval_step(idx)
        trainer.history['slice_idx'].append(idx)
        trainer.history['dice'].append(res['dice'])
        trainer.history['n_masks'].append(res['n_masks'])
        print(f"Slice {idx} | Dice: {res['dice']:.4f}")

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(trainer.history)
    plot_clean_results(df, out_dir / 'precision_dice.png')
    with open(out_dir / 'config_supervised.json', 'w') as f:
        json.dump({'params': best_p, 'mean_dice': df['dice'].mean()}, f, indent=2, cls=NumpyEncoder)

if __name__ == "__main__":
    main()