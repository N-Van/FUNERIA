"""
Expérience 4: Comparaison IoU de toutes les méthodes
Compare grid points, bounding boxes, YOLO+threshold, et 2.5D
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import Config
from utils import apply_clahe, convert_rgb, get_urna_mask_threshold, match_masks_between_slices, make_grid_boxes
from sam_inference import sam_inference, sam_inference_boxes


def load_models():
    """Charge tous les modèles nécessaires"""
    from ultralytics import SAM, YOLO
    
    print("Chargement des modèles...")
    model_sam = SAM(Config.SAM_WEIGHTS)
    model_yolo = YOLO(Config.YOLO_WEIGHTS)
    
    return model_sam, model_yolo


def create_25d_image_with_clahe(volume, slice_idx, use_clahe=True):
    """Crée une image 2.5D"""
    D, H, W = volume.shape
    idx_prev = max(0, slice_idx - 1)
    idx_curr = slice_idx
    idx_next = min(D - 1, slice_idx + 1)
    
    slice_prev = apply_clahe(volume[idx_prev], use_clahe=use_clahe)
    slice_curr = apply_clahe(volume[idx_curr], use_clahe=use_clahe)
    slice_next = apply_clahe(volume[idx_next], use_clahe=use_clahe)
    
    img_25d = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
    return img_25d


def detect_urn_with_yolo(model_yolo, image, conf=0.25):
    """Détecte l'urne avec YOLO"""
    from utils import normalize_to_uint8
    
    img = normalize_to_uint8(image)
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    
    results = model_yolo.predict(source=img, conf=conf, verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, np.ones(image.shape[:2], dtype=bool)
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    bbox = boxes[largest_idx].astype(int)
    
    mask_roi = np.zeros(image.shape[:2], dtype=bool)
    x1, y1, x2, y2 = bbox
    mask_roi[y1:y2, x1:x2] = True
    
    return bbox, mask_roi


def combine_yolo_and_threshold_masks(yolo_mask, threshold_mask):
    """Combine YOLO et seuillage"""
    if yolo_mask is None:
        return threshold_mask
    if threshold_mask is None:
        return yolo_mask
    return np.logical_and(yolo_mask, threshold_mask)


def run_all_iou_analyses(model_sam, model_yolo, vol, mid, slice_range=10):
    """Lance toutes les analyses IoU"""
    
    results = {}
    slice_beginning = mid - slice_range // 2
    slice_end = mid + slice_range // 2
    
    # Générer boxes une fois
    H_vol, W_vol = vol[0].shape
    boxes = make_grid_boxes(H_vol, W_vol, Config.GRID_STRIDE, 64)
    
    # 1. Grid Points
    print("\n" + "="*60)
    print("1/4 - ANALYSE IoU: GRID POINTS")
    print("="*60)
    
    iou_grid = []
    for i in range(slice_beginning, slice_end):
        print(f"  Slice {i - slice_beginning + 1}/{slice_end - slice_beginning}")
        
        sl_prev = apply_clahe(vol[i])
        sl_curr = apply_clahe(vol[i + 1])
        img_rgb_prev = convert_rgb(sl_prev)
        img_rgb_curr = convert_rgb(sl_curr)
        urna_mask_prev = get_urna_mask_threshold(sl_prev)
        urna_mask_curr = get_urna_mask_threshold(sl_curr)
        
        masks_prev, _, _ = sam_inference(model_sam, img_rgb_prev, urna_mask_prev, mode="grid", make_colored=False)
        masks_curr, _, _ = sam_inference(model_sam, img_rgb_curr, urna_mask_curr, mode="grid", make_colored=False)
        
        iou = match_masks_between_slices(masks_prev, masks_curr)
        iou_grid.append(iou)
        
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    results['grid'] = iou_grid
    print(f"Moyenne IoU (grid): {np.mean(iou_grid):.4f}")
    
    # 2. Bounding Boxes
    print("\n" + "="*60)
    print("2/4 - ANALYSE IoU: BOUNDING BOXES")
    print("="*60)
    
    iou_boxes = []
    for i in range(slice_beginning, slice_end):
        print(f"  Slice {i - slice_beginning + 1}/{slice_end - slice_beginning}")
        
        sl_prev = apply_clahe(vol[i])
        sl_curr = apply_clahe(vol[i + 1])
        img_rgb_prev = convert_rgb(sl_prev)
        img_rgb_curr = convert_rgb(sl_curr)
        urna_mask_prev = get_urna_mask_threshold(sl_prev)
        urna_mask_curr = get_urna_mask_threshold(sl_curr)
        
        masks_prev, _, _ = sam_inference_boxes(model_sam, img_rgb_prev, urna_mask_prev, boxes, make_colored=False)
        masks_curr, _, _ = sam_inference_boxes(model_sam, img_rgb_curr, urna_mask_curr, boxes, make_colored=False)
        
        iou = match_masks_between_slices(masks_prev, masks_curr)
        iou_boxes.append(iou)
        
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    results['boxes'] = iou_boxes
    print(f"Moyenne IoU (boxes): {np.mean(iou_boxes):.4f}")
    
    # 3. YOLO + Threshold
    print("\n" + "="*60)
    print("3/4 - ANALYSE IoU: YOLO + THRESHOLD")
    print("="*60)
    
    iou_yolo = []
    for i in range(slice_beginning, slice_end):
        print(f"  Slice {i - slice_beginning + 1}/{slice_end - slice_beginning}")
        
        sl_prev = apply_clahe(vol[i])
        sl_curr = apply_clahe(vol[i + 1])
        img_rgb_prev = convert_rgb(sl_prev)
        img_rgb_curr = convert_rgb(sl_curr)
        
        # Masques YOLO + threshold
        _, yolo_mask_prev = detect_urn_with_yolo(model_yolo, sl_prev)
        _, yolo_mask_curr = detect_urn_with_yolo(model_yolo, sl_curr)
        threshold_mask_prev = get_urna_mask_threshold(sl_prev)
        threshold_mask_curr = get_urna_mask_threshold(sl_curr)
        urna_mask_prev = combine_yolo_and_threshold_masks(yolo_mask_prev, threshold_mask_prev)
        urna_mask_curr = combine_yolo_and_threshold_masks(yolo_mask_curr, threshold_mask_curr)
        
        masks_prev, _, _ = sam_inference(model_sam, img_rgb_prev, urna_mask_prev, mode="grid", make_colored=False)
        masks_curr, _, _ = sam_inference(model_sam, img_rgb_curr, urna_mask_curr, mode="grid", make_colored=False)
        
        iou = match_masks_between_slices(masks_prev, masks_curr)
        iou_yolo.append(iou)
        
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    results['yolo'] = iou_yolo
    print(f"Moyenne IoU (yolo): {np.mean(iou_yolo):.4f}")
    
    # 4. 2.5D
    print("\n" + "="*60)
    print("4/4 - ANALYSE IoU: 2.5D")
    print("="*60)
    
    # Référence
    sl_ref = apply_clahe(vol[slice_beginning])
    urna_mask_ref = get_urna_mask_threshold(sl_ref)
    img_25d_ref = create_25d_image_with_clahe(vol, slice_beginning)
    
    with torch.no_grad():
        masks_ref, _, _ = sam_inference(model_sam, img_25d_ref, urna_mask_ref, mode="grid", make_colored=False)
        if isinstance(masks_ref, torch.Tensor):
            masks_ref = masks_ref.cpu().numpy()
    
    torch.cuda.empty_cache()
    
    iou_25d = []
    for i in range(slice_beginning + 1, slice_end + 1):
        print(f"  Comparaison slice {slice_beginning} vs {i}")
        
        sl_curr = apply_clahe(vol[i])
        urna_mask_curr = get_urna_mask_threshold(sl_curr)
        img_25d_curr = create_25d_image_with_clahe(vol, i)
        
        with torch.no_grad():
            masks_curr, _, _ = sam_inference(model_sam, img_25d_curr, urna_mask_curr, mode="grid", make_colored=False)
            if isinstance(masks_curr, torch.Tensor):
                masks_curr = masks_curr.cpu().numpy()
        
        iou = match_masks_between_slices(masks_ref, masks_curr)
        iou_25d.append(iou)
        
        del masks_curr
        torch.cuda.empty_cache()
    
    results['25d'] = iou_25d
    print(f"Moyenne IoU (2.5D): {np.mean(iou_25d):.4f}")
    
    del masks_ref
    torch.cuda.empty_cache()
    
    return results


def visualize_all_comparisons(results, output_dir):
    """Visualise toutes les comparaisons"""
    
    # Calcul des moyennes
    means = {
        'Grid Points': np.mean(results['grid']),
        'Bounding Boxes': np.mean(results['boxes']),
        'YOLO + Threshold': np.mean(results['yolo']),
        '2.5D': np.mean(results['25d'])
    }
    
    # Affichage texte
    print("\n" + "="*70)
    print("COMPARAISON FINALE - TOUTES LES MÉTHODES")
    print("="*70)
    for method, mean_iou in means.items():
        print(f"{method:20s}: {mean_iou:.4f}")
    print("="*70)
    
    # Plot 1: IoU par paire de slices
    plt.figure(figsize=(14, 8))
    x_axis = range(1, len(results['grid']) + 1)
    
    plt.plot(x_axis, results['grid'], marker='o', label=f"Grid Points (moy={means['Grid Points']:.4f})", linewidth=2)
    plt.plot(x_axis, results['boxes'], marker='s', label=f"Bounding Boxes (moy={means['Bounding Boxes']:.4f})", linewidth=2)
    plt.plot(x_axis, results['yolo'], marker='^', label=f"YOLO + Threshold (moy={means['YOLO + Threshold']:.4f})", linewidth=2)
    
    # 2.5D a potentiellement moins de points
    x_25d = range(1, len(results['25d']) + 1)
    plt.plot(x_25d, results['25d'], marker='D', label=f"2.5D (moy={means['2.5D']:.4f})", linewidth=2)
    
    plt.xlabel("Index de paire de slices", fontsize=12)
    plt.ylabel("IoU", fontsize=12)
    plt.title("Comparaison IoU entre slices consécutives\n(Toutes les méthodes)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path / "exp4_all_methods_iou.png", dpi=150)
    print(f"\nGraphique sauvegardé: {output_path / 'exp4_all_methods_iou.png'}")
    plt.show()
    
    # Plot 2: Barres de comparaison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(means.keys())
    values = list(means.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(methods, values, color=colors, alpha=0.7)
    ax.set_ylabel('IoU moyen', fontsize=12)
    ax.set_title('Comparaison des moyennes IoU par méthode', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter valeurs sur barres
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / "exp4_methods_comparison_bars.png", dpi=150)
    print(f"Graphique sauvegardé: {output_path / 'exp4_methods_comparison_bars.png'}")
    plt.show()
    
    return means


def main():
    """Fonction principale"""
    
    # Charger volume
    print("Chargement du volume...")
    vol = tifffile.imread(Config.TIF_PATH)
    print(f"Volume: shape={vol.shape}, dtype={vol.dtype}")
    
    mid = len(vol) // 2
    
    # Charger modèles
    model_sam, model_yolo = load_models()
    
    # Lancer toutes les analyses
    results = run_all_iou_analyses(model_sam, model_yolo, vol, mid, Config.SLICE_RANGE)
    
    # Visualisations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(Config.OUTPUT_DIR) / f"exp4_comparison_{timestamp}"
    means = visualize_all_comparisons(results, output_dir)
    
    # Sauvegarder résultats
    np.savez(
        output_dir / "results.npz",
        iou_grid=results['grid'],
        iou_boxes=results['boxes'],
        iou_yolo=results['yolo'],
        iou_25d=results['25d'],
        mean_grid=means['Grid Points'],
        mean_boxes=means['Bounding Boxes'],
        mean_yolo=means['YOLO + Threshold'],
        mean_25d=means['2.5D']
    )
    
    # Créer tableau récapitulatif
    df_summary = pd.DataFrame({
        'Méthode': list(means.keys()),
        'IoU Moyen': list(means.values())
    })
    df_summary = df_summary.sort_values('IoU Moyen', ascending=False)
    df_summary.to_csv(output_dir / "summary.csv", index=False)
    
    print(f"\n{'='*70}")
    print(f"RÉSULTATS SAUVEGARDÉS DANS: {output_dir}")
    print(f"{'='*70}")
    print(f"\nFichiers générés:")
    print(f"  - results.npz: données brutes")
    print(f"  - summary.csv: tableau récapitulatif")
    print(f"  - exp4_all_methods_iou.png: graphique courbes")
    print(f"  - exp4_methods_comparison_bars.png: graphique barres")


if __name__ == "__main__":
    main()
