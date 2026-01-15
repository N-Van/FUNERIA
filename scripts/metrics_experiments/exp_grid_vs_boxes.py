"""
Expérience 1: Comparaison Grid Points vs Bounding Boxes
Compare les performances IoU des deux méthodes de prompting
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile
from pathlib import Path
from datetime import datetime

from config import Config
from utils import apply_clahe, convert_rgb, get_urna_mask_threshold, match_masks_between_slices, make_grid_boxes
from sam_inference import sam_inference, sam_inference_boxes


def run_iou_analysis_grid(model, vol, mid, slice_range=10):
    """Analyse IoU avec grille de points"""
    print("\n" + "="*60)
    print("ANALYSE IoU AVEC GRILLE DE POINTS")
    print("="*60)
    
    slice_beginning = mid - slice_range // 2
    slice_end = mid + slice_range // 2
    iou_values = []

    for i in range(slice_beginning, slice_end):
        print(f"Slice {i - slice_beginning + 1}/{slice_end - slice_beginning}")

        # Slice i
        sl_prev = vol[i]
        sl_prev = apply_clahe(sl_prev)
        img_rgb_prev = convert_rgb(sl_prev)
        urna_mask_prev = get_urna_mask_threshold(sl_prev)

        # Slice i+1
        sl_curr = vol[i + 1]
        sl_curr = apply_clahe(sl_curr)
        img_rgb_curr = convert_rgb(sl_curr)
        urna_mask_curr = get_urna_mask_threshold(sl_curr)

        # Inférence
        masks_prev, _, _ = sam_inference(
            model,
            img_rgb=img_rgb_prev,
            urna_mask=urna_mask_prev,
            mode="grid",
            make_colored=False
        )

        masks_curr, _, _ = sam_inference(
            model,
            img_rgb=img_rgb_curr,
            urna_mask=urna_mask_curr,
            mode="grid",
            make_colored=False
        )

        # IoU
        iou = match_masks_between_slices(masks_prev, masks_curr)
        iou_values.append(iou)
        print(f"  IoU: {iou:.4f}")

        # Libérer mémoire
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
            del masks_prev, masks_curr, img_rgb_prev, img_rgb_curr
            torch.cuda.empty_cache()

    print(f"\nMoyenne IoU (grid): {np.mean(iou_values):.4f}")
    return iou_values


def run_iou_analysis_boxes(model, vol, mid, slice_range=10):
    """Analyse IoU avec bounding boxes"""
    print("\n" + "="*60)
    print("ANALYSE IoU AVEC BOUNDING BOXES")
    print("="*60)
    
    slice_beginning = mid - slice_range // 2
    slice_end = mid + slice_range // 2
    iou_values = []

    # Générer boxes une fois
    H_vol, W_vol = vol[0].shape
    boxes = make_grid_boxes(H_vol, W_vol, Config.GRID_STRIDE, 64)

    for i in range(slice_beginning, slice_end):
        print(f"Slice {i - slice_beginning + 1}/{slice_end - slice_beginning}")

        # Slice i
        sl_prev = vol[i]
        sl_prev = apply_clahe(sl_prev)
        img_rgb_prev = convert_rgb(sl_prev)
        urna_mask_prev = get_urna_mask_threshold(sl_prev)

        # Slice i+1
        sl_curr = vol[i + 1]
        sl_curr = apply_clahe(sl_curr)
        img_rgb_curr = convert_rgb(sl_curr)
        urna_mask_curr = get_urna_mask_threshold(sl_curr)

        # Inférence
        masks_prev, _, _ = sam_inference_boxes(
            model,
            img_rgb=img_rgb_prev,
            urna_mask=urna_mask_prev,
            boxes=boxes,
            make_colored=False
        )

        masks_curr, _, _ = sam_inference_boxes(
            model,
            img_rgb=img_rgb_curr,
            urna_mask=urna_mask_curr,
            boxes=boxes,
            make_colored=False
        )

        # IoU
        iou = match_masks_between_slices(masks_prev, masks_curr)
        iou_values.append(iou)
        print(f"  IoU: {iou:.4f}")

        # Libérer mémoire
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
            del masks_prev, masks_curr, img_rgb_prev, img_rgb_curr
            torch.cuda.empty_cache()

    print(f"\nMoyenne IoU (boxes): {np.mean(iou_values):.4f}")
    return iou_values


def visualize_comparison(iou_grid, iou_boxes, output_dir):
    """Visualise la comparaison des IoU"""
    mean_grid = np.mean(iou_grid)
    mean_boxes = np.mean(iou_boxes)

    print("\n" + "="*60)
    print("COMPARAISON FINALE")
    print("="*60)
    print(f"Moyenne IoU (grille de points): {mean_grid:.4f}")
    print(f"Moyenne IoU (bounding boxes):   {mean_boxes:.4f}")
    print(f"Différence: {abs(mean_grid - mean_boxes):.4f}")
    print("="*60)

    # Plot
    plt.figure(figsize=(10, 6))
    x_axis = range(1, len(iou_grid) + 1)
    plt.plot(x_axis, iou_grid, marker='o', label=f"Grille de points (moy={mean_grid:.4f})", linewidth=2)
    plt.plot(x_axis, iou_boxes, marker='s', label=f"Bounding boxes (moy={mean_boxes:.4f})", linewidth=2)
    plt.xlabel("Index de paire de slices", fontsize=12)
    plt.ylabel("IoU", fontsize=12)
    plt.title("Comparaison IoU entre slices consécutives\n(Grille de points vs Bounding boxes)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path / "exp1_grid_vs_boxes.png", dpi=150)
    print(f"\nGraphique sauvegardé: {output_path / 'exp1_grid_vs_boxes.png'}")
    plt.show()


def main():
    """Fonction principale"""
    from ultralytics import SAM
    
    # Charger volume
    print("Chargement du volume...")
    vol = tifffile.imread(Config.TIF_PATH)
    print(f"Volume: shape={vol.shape}, dtype={vol.dtype}")
    
    mid = len(vol) // 2
    
    # Charger modèle
    print(f"\nChargement du modèle SAM ({Config.SAM_WEIGHTS})...")
    model = SAM(Config.SAM_WEIGHTS)
    
    # Analyses IoU
    iou_grid = run_iou_analysis_grid(model, vol, mid, Config.SLICE_RANGE)
    iou_boxes = run_iou_analysis_boxes(model, vol, mid, Config.SLICE_RANGE)
    
    # Visualisation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(Config.OUTPUT_DIR) / f"exp1_grid_vs_boxes_{timestamp}"
    visualize_comparison(iou_grid, iou_boxes, output_dir)
    
    # Sauvegarder résultats
    np.savez(
        output_dir / "results.npz",
        iou_grid=iou_grid,
        iou_boxes=iou_boxes,
        mean_grid=np.mean(iou_grid),
        mean_boxes=np.mean(iou_boxes)
    )
    print(f"\nRésultats sauvegardés: {output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
