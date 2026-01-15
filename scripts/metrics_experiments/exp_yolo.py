"""
Expérience 2: Détection YOLO pour localisation de l'urne
Compare YOLO seul, seuillage seul, et combinaison des deux
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile
from pathlib import Path
from datetime import datetime

from config import Config
from utils import apply_clahe, convert_rgb, get_urna_mask_threshold, normalize_to_uint8, match_masks_between_slices
from sam_inference import sam_inference


def detect_urn_with_yolo(model_yolo, image, conf=0.25):
    """
    Détecte l'urne avec YOLO et retourne une bbox et un masque ROI
    """
    # Normaliser l'image
    img = normalize_to_uint8(image)
    if img.ndim == 2:
        img = np.dstack([img, img, img])

    # Détection YOLO
    results = model_yolo.predict(source=img, conf=conf, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        print("[YOLO] Aucun objet détecté, utilisation de l'image complète")
        return None, np.ones(image.shape[:2], dtype=bool)

    # Prendre la plus grande bbox
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    bbox = boxes[largest_idx].astype(int)

    # Créer un masque ROI
    mask_roi = np.zeros(image.shape[:2], dtype=bool)
    x1, y1, x2, y2 = bbox
    mask_roi[y1:y2, x1:x2] = True

    print(f"[YOLO] Urne détectée à [{x1}, {y1}, {x2}, {y2}]")
    return bbox, mask_roi


def combine_yolo_and_threshold_masks(yolo_mask, threshold_mask):
    """Combine le masque YOLO avec le masque de seuillage"""
    if yolo_mask is None:
        return threshold_mask
    if threshold_mask is None:
        return yolo_mask

    combined = np.logical_and(yolo_mask, threshold_mask)
    print(f"[Combinaison] YOLO: {yolo_mask.sum()} pixels, Seuillage: {threshold_mask.sum()} pixels, Combiné: {combined.sum()} pixels")
    return combined


def sam_inference_with_yolo(
    model,
    model_yolo,
    img_rgb,
    use_yolo=True,
    use_threshold=True,
    threshold_value=None,
    yolo_conf=0.25,
    mode="grid",
    **sam_params
):
    """
    Inférence SAM avec combinaison optionnelle YOLO + seuillage
    """
    if threshold_value is None:
        threshold_value = Config.URNA_THRESHOLD
    
    H, W, _ = img_rgb.shape
    img_gray = img_rgb[..., :3].mean(axis=2).astype(np.uint8)

    # Créer les masques
    yolo_mask = None
    threshold_mask = None
    bbox_yolo = None

    if use_yolo:
        bbox_yolo, yolo_mask = detect_urn_with_yolo(model_yolo, img_gray, conf=yolo_conf)

    if use_threshold:
        threshold_mask = get_urna_mask_threshold(img_gray, thr=threshold_value)

    # Combiner les masques
    if use_yolo and use_threshold:
        urna_mask = combine_yolo_and_threshold_masks(yolo_mask, threshold_mask)
    elif use_yolo:
        urna_mask = yolo_mask
    elif use_threshold:
        urna_mask = threshold_mask
    else:
        urna_mask = None

    # Inférence SAM
    masks, colored, info = sam_inference(
        model,
        img_rgb=img_rgb,
        urna_mask=urna_mask,
        mode=mode,
        **sam_params
    )

    # Ajouter infos YOLO
    info['bbox_yolo'] = bbox_yolo
    info['use_yolo'] = use_yolo
    info['use_threshold'] = use_threshold

    return masks, colored, info, bbox_yolo


def test_yolo_methods(model, model_yolo, vol, mid):
    """Teste les 3 méthodes de détection"""
    print("\n" + "="*60)
    print("TEST SEGMENTATION AVEC YOLO + SEUILLAGE")
    print("="*60)

    # Préparer l'image
    sl = vol[mid]
    sl = apply_clahe(sl)
    img_rgb = convert_rgb(sl)

    results = {}

    # Test 1: YOLO seul
    print("\n--- Test 1: YOLO seul ---")
    masks_yolo, colored_yolo, info_yolo, bbox_yolo = sam_inference_with_yolo(
        model,
        model_yolo,
        img_rgb=img_rgb,
        use_yolo=True,
        use_threshold=False,
    )
    results['yolo'] = {
        'masks': masks_yolo,
        'colored': colored_yolo,
        'info': info_yolo,
        'bbox': bbox_yolo
    }
    print(f"Résultat: {info_yolo['n_masks_filtered']} masques")

    # Test 2: Seuillage seul
    print("\n--- Test 2: Seuillage seul ---")
    masks_thr, colored_thr, info_thr, _ = sam_inference_with_yolo(
        model,
        model_yolo,
        img_rgb=img_rgb,
        use_yolo=False,
        use_threshold=True,
    )
    results['threshold'] = {
        'masks': masks_thr,
        'colored': colored_thr,
        'info': info_thr,
        'bbox': None
    }
    print(f"Résultat: {info_thr['n_masks_filtered']} masques")

    # Test 3: YOLO + Seuillage
    print("\n--- Test 3: YOLO + Seuillage combinés ---")
    masks_comb, colored_comb, info_comb, bbox_comb = sam_inference_with_yolo(
        model,
        model_yolo,
        img_rgb=img_rgb,
        use_yolo=True,
        use_threshold=True,
    )
    results['combined'] = {
        'masks': masks_comb,
        'colored': colored_comb,
        'info': info_comb,
        'bbox': bbox_comb
    }
    print(f"Résultat: {info_comb['n_masks_filtered']} masques")

    return results, img_rgb


def run_iou_analysis_yolo(model, model_yolo, vol, mid, slice_range=10):
    """Analyse IoU avec YOLO + seuillage"""
    print("\n" + "="*60)
    print("ANALYSE IoU AVEC YOLO + SEUILLAGE")
    print("="*60)

    slice_beginning = mid - slice_range // 2
    slice_end = mid + slice_range // 2
    iou_values = []

    for i in range(slice_beginning, slice_end):
        print(f"Slice {i - slice_beginning + 1}/{slice_end - slice_beginning}")

        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Slice i
        sl_prev = vol[i]
        sl_prev = apply_clahe(sl_prev)
        img_rgb_prev = convert_rgb(sl_prev)

        # Slice i+1
        sl_curr = vol[i + 1]
        sl_curr = apply_clahe(sl_curr)
        img_rgb_curr = convert_rgb(sl_curr)

        # Inférence
        masks_prev, _, _, _ = sam_inference_with_yolo(
            model,
            model_yolo,
            img_rgb=img_rgb_prev,
            use_yolo=True,
            use_threshold=True,
            make_colored=False
        )

        masks_curr, _, _, _ = sam_inference_with_yolo(
            model,
            model_yolo,
            img_rgb=img_rgb_curr,
            use_yolo=True,
            use_threshold=True,
            make_colored=False
        )

        # IoU
        iou = match_masks_between_slices(masks_prev, masks_curr)
        iou_values.append(iou)
        print(f"  IoU: {iou:.4f}")

        # Libérer mémoire
        if Config.DEVICE == "cuda":
            del masks_prev, masks_curr, img_rgb_prev, img_rgb_curr, sl_prev, sl_curr
            torch.cuda.empty_cache()

    print(f"\nMoyenne IoU (YOLO + seuillage): {np.mean(iou_values):.4f}")
    return iou_values


def visualize_yolo_methods(results, img_original, output_dir):
    """Visualise les 3 méthodes"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Image originale
    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis('off')

    # YOLO seul
    axes[0, 1].imshow(results['yolo']['colored'])
    if results['yolo']['bbox'] is not None:
        x1, y1, x2, y2 = results['yolo']['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f"YOLO seul ({results['yolo']['info']['n_masks_filtered']} masques)")
    axes[0, 1].axis('off')

    # Seuillage seul
    axes[1, 0].imshow(results['threshold']['colored'])
    axes[1, 0].set_title(f"Seuillage seul ({results['threshold']['info']['n_masks_filtered']} masques)")
    axes[1, 0].axis('off')

    # YOLO + Seuillage
    axes[1, 1].imshow(results['combined']['colored'])
    if results['combined']['bbox'] is not None:
        x1, y1, x2, y2 = results['combined']['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
        axes[1, 1].add_patch(rect)
    axes[1, 1].set_title(f"YOLO + Seuillage ({results['combined']['info']['n_masks_filtered']} masques)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path / "exp2_yolo_methods.png", dpi=150)
    print(f"\nGraphique sauvegardé: {output_path / 'exp2_yolo_methods.png'}")
    plt.show()


def main():
    """Fonction principale"""
    from ultralytics import SAM, YOLO
    
    # Charger volume
    print("Chargement du volume...")
    vol = tifffile.imread(Config.TIF_PATH)
    print(f"Volume: shape={vol.shape}, dtype={vol.dtype}")
    
    mid = len(vol) // 2
    
    # Charger modèles
    print(f"\nChargement des modèles...")
    model = SAM(Config.SAM_WEIGHTS)
    model_yolo = YOLO(Config.YOLO_WEIGHTS)
    
    # Tests des méthodes
    results, img_original = test_yolo_methods(model, model_yolo, vol, mid)
    
    # Analyse IoU
    iou_yolo = run_iou_analysis_yolo(model, model_yolo, vol, mid, Config.SLICE_RANGE)
    
    # Visualisation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(Config.OUTPUT_DIR) / f"exp2_yolo_{timestamp}"
    visualize_yolo_methods(results, img_original, output_dir)
    
    # Sauvegarder résultats
    np.savez(
        output_dir / "results.npz",
        iou_yolo=iou_yolo,
        mean_iou=np.mean(iou_yolo),
        n_masks_yolo=results['yolo']['info']['n_masks_filtered'],
        n_masks_threshold=results['threshold']['info']['n_masks_filtered'],
        n_masks_combined=results['combined']['info']['n_masks_filtered']
    )
    print(f"\nRésultats sauvegardés: {output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
