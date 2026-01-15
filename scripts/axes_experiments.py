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
import argparse
import tifffile

def normalize_to_uint8(img):
    mn, mx = float(img.min()), float(img.max())
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)

def apply_clahe(sl):
    sl_prep = normalize_to_uint8(sl)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(sl_prep)

def compute_dice_score(pred_masks, gt_mask):
    if not pred_masks or gt_mask.sum() == 0:
        return 0.0
    full_pred = np.logical_or.reduce(pred_masks)
    binary_gt = (gt_mask > 0)
    intersection = np.logical_and(full_pred, binary_gt).sum()
    total = full_pred.sum() + binary_gt.sum()
    return (2.0 * intersection) / total if total > 0 else 1.0

def sam_inference(model, img, urna_mask, params):
    H, W = img.shape[:2]
    img_rgb = np.repeat(img[..., np.newaxis], 3, -1) if img.ndim == 2 else img
    stride = params['grid_stride']
    xs = np.arange(stride // 2, W, stride)
    ys = np.arange(stride // 2, H, stride)
    pts = [(int(x), int(y)) for y in ys for x in xs if urna_mask[y, x]]
    
    if not pts: return [], 0
    
    all_masks = []
    batch_size = 40 # Limite la charge mémoire
    
    for i in range(0, len(pts), batch_size):
        batch_pts = pts[i : i + batch_size]
        with torch.inference_mode():
            results = model(img_rgb, points=batch_pts, labels=[1]*len(batch_pts), 
                            device="cuda", conf=params['conf'], iou=0.25, 
                            max_det=50, show=False, verbose=False)
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for m in masks:
                if 300 <= (m > 0).sum() <= (H*W*0.05):
                    all_masks.append(m > 0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return all_masks, len(all_masks)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def run_axis_experiment(model, vol, gt, params, n_steps=10):
    indices = np.linspace(0, vol.shape[0]-1, n_steps, dtype=int)
    scores = []
    for idx in indices:
        clear_gpu_memory()
        sl = apply_clahe(vol[idx])
        urna = (sl >= 60).astype(np.uint8) * 255
        preds, _ = sam_inference(model, sl, urna, params)
        dice = compute_dice_score(preds, gt[idx])
        scores.append(dice)
    return np.mean(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--model', default='sam_b.pt')
    parser.add_argument('--output', default='./results_axes')
    args = parser.parse_args()

    vol_orig = tifffile.imread(args.volume)
    gt_orig = tifffile.imread(args.gt)
    
    if vol_orig.shape[2] < vol_orig.shape[0]: vol_orig = np.transpose(vol_orig, (2, 0, 1))
    if gt_orig.shape[2] < gt_orig.shape[0]: gt_orig = np.transpose(gt_orig, (2, 0, 1))

    from ultralytics import SAM
    model = SAM(args.model)
    params = {'grid_stride': 80, 'conf': 0.45}
    results = {}

    print("Running Axial (Z) axis experiment...")
    results['Axial (Z)'] = run_axis_experiment(model, vol_orig, gt_orig, params)

    print("Running Coronal (Y) axis experiment...")
    vol_cor = np.transpose(vol_orig, (1, 0, 2))
    gt_cor = np.transpose(gt_orig, (1, 0, 2))
    results['Coronal (Y)'] = run_axis_experiment(model, vol_cor, gt_cor, params)

    print("Running Sagittal (X) axis experiment...")
    vol_sag = np.transpose(vol_orig, (2, 0, 1))
    gt_sag = np.transpose(gt_orig, (2, 0, 1))
    results['Sagittal (X)'] = run_axis_experiment(model, vol_sag, gt_sag, params)

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    
    bars = plt.bar(names, values, color=['#2980b9', '#d35400', '#27ae60'], alpha=0.8)
    plt.ylabel('Dice Score Moyen')
    plt.title('Performance SAM par Axe de Coupe')
    plt.ylim(0, 1.05)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha='center', fontweight='bold')
    
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / 'comparaison_axes.png', dpi=150)
    
    with open(out_dir / 'axes_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()