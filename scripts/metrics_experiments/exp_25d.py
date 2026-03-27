"""
Expérience 3: Segmentation 2.5D (utilisation de slices adjacentes)
Compare 2D classique vs 2.5D et teste différents paramètres
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import Config
from utils import apply_clahe, convert_rgb, get_urna_mask_threshold, match_masks_between_slices
from sam_inference import sam_inference


def create_25d_image_with_clahe(volume, slice_idx, use_clahe=True):
    """Crée une image 2.5D avec slices adjacentes"""
    D, H, W = volume.shape

    # Gérer les cas limites
    idx_prev = max(0, slice_idx - 1)
    idx_curr = slice_idx
    idx_next = min(D - 1, slice_idx + 1)

    # Extraire et prétraiter les slices
    slice_prev = apply_clahe(volume[idx_prev], use_clahe=use_clahe)
    slice_curr = apply_clahe(volume[idx_curr], use_clahe=use_clahe)
    slice_next = apply_clahe(volume[idx_next], use_clahe=use_clahe)

    # Empiler dans les canaux RGB
    img_25d = np.stack([slice_prev, slice_curr, slice_next], axis=-1)

    return img_25d


def sam_inference_25d(
    model,
    volume,
    slice_idx,
    urna_mask=None,
    use_clahe=True,
    mode="grid",
    **sam_params
):
    """Inférence SAM avec image 2.5D"""
    # Créer l'image 2.5D
    img_25d = create_25d_image_with_clahe(volume, slice_idx, use_clahe=use_clahe)

    print(f"[2.5D] Image créée: shape={img_25d.shape}, dtype={img_25d.dtype}")
    print(f"       Slices utilisées: [{max(0, slice_idx-1)}, {slice_idx}, {min(len(volume)-1, slice_idx+1)}]")

    # Inférence SAM
    masks, colored, info = sam_inference(
        model,
        img_rgb=img_25d,
        urna_mask=urna_mask,
        mode=mode,
        **sam_params
    )

    # Ajouter infos
    info['mode_25d'] = True
    info['slice_idx'] = slice_idx

    return masks, colored, info, img_25d


def compare_2d_vs_25d(model, vol, mid):
    """Compare segmentation 2D vs 2.5D sur slice du milieu"""
    print("\n" + "="*60)
    print("COMPARAISON SEGMENTATION 2D vs 2.5D")
    print("="*60)

    # Préparer masque urne
    sl_mid = vol[mid]
    sl_mid_clahe = apply_clahe(sl_mid)
    urna_mask = get_urna_mask_threshold(sl_mid)

    # Test 1: Segmentation 2D
    print("\n--- Test 1: Segmentation 2D classique ---")
    img_rgb_2d = convert_rgb(sl_mid_clahe)
    masks_2d, colored_2d, info_2d = sam_inference(
        model,
        img_rgb=img_rgb_2d,
        urna_mask=urna_mask,
        mode="grid"
    )
    print(f"Résultat 2D: {info_2d['n_masks_filtered']} masques")

    # Test 2: Segmentation 2.5D
    print("\n--- Test 2: Segmentation 2.5D ---")
    masks_25d, colored_25d, info_25d, img_25d = sam_inference_25d(
        model,
        volume=vol,
        slice_idx=mid,
        urna_mask=urna_mask,
        use_clahe=True,
        mode="grid"
    )
    print(f"Résultat 2.5D: {info_25d['n_masks_filtered']} masques")

    # Libérer mémoire
    if Config.DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        '2d': {'masks': masks_2d, 'colored': colored_2d, 'info': info_2d, 'img': img_rgb_2d},
        '25d': {'masks': masks_25d, 'colored': colored_25d, 'info': info_25d, 'img': img_25d},
        'urna_mask': urna_mask
    }


def run_iou_analysis_25d(model, vol, mid, num_slices=10):
    """Analyse IoU avec approche 2.5D (slice fixe)"""
    print("\n" + "="*60)
    print("ANALYSE IoU AVEC SEGMENTATION 2.5D")
    print("="*60)

    slice_start = mid
    iou_values = []

    # Nettoyer avant
    torch.cuda.empty_cache()

    # Préparer slice de référence
    sl_ref_2d = apply_clahe(vol[slice_start])
    urna_mask_ref = get_urna_mask_threshold(sl_ref_2d)

    # Inférence sur référence
    with torch.no_grad():
        masks_ref, _, _, _ = sam_inference_25d(
            model,
            volume=vol,
            slice_idx=slice_start,
            urna_mask=urna_mask_ref,
            use_clahe=True,
            mode="grid",
            make_colored=False
        )

        if isinstance(masks_ref, torch.Tensor):
            masks_ref = masks_ref.cpu().numpy()

    torch.cuda.empty_cache()

    # Comparer avec slices suivantes
    for i in range(slice_start + 1, slice_start + 1 + num_slices):
        print(f"Comparaison slice {slice_start} vs slice {i}")

        torch.cuda.empty_cache()

        # Préparer slice courante
        sl_curr_2d = apply_clahe(vol[i])
        urna_mask_curr = get_urna_mask_threshold(sl_curr_2d)

        with torch.no_grad():
            masks_curr, _, _, _ = sam_inference_25d(
                model,
                volume=vol,
                slice_idx=i,
                urna_mask=urna_mask_curr,
                use_clahe=True,
                mode="grid",
                make_colored=False
            )

            if isinstance(masks_curr, torch.Tensor):
                masks_curr = masks_curr.cpu().numpy()

        # Calculer IoU
        iou = match_masks_between_slices(masks_ref, masks_curr)
        iou_values.append(iou)
        print(f"  IoU: {iou:.4f}")

        # Libérer
        del masks_curr, sl_curr_2d, urna_mask_curr
        torch.cuda.empty_cache()

    print(f"\nMoyenne IoU (2.5D): {np.mean(iou_values):.4f}")

    # Nettoyage final
    del masks_ref, sl_ref_2d, urna_mask_ref
    torch.cuda.empty_cache()

    return iou_values


def test_delta_z_variations(model, vol, mid):
    """Teste différentes valeurs de delta_z"""
    print("\n" + "="*60)
    print("TEST VARIATIONS DELTA_Z")
    print("="*60)

    sl = vol[mid]
    urna_mask = get_urna_mask_threshold(sl)

    results = []

    for delta_z in Config.DELTA_Z_VALUES:
        print(f"\nTest delta_z = {delta_z}")

        # Créer image avec espacement
        D, H, W = vol.shape
        idx_prev = max(0, mid - delta_z)
        idx_curr = mid
        idx_next = min(D - 1, mid + delta_z)

        slice_prev = apply_clahe(vol[idx_prev])
        slice_curr = apply_clahe(vol[idx_curr])
        slice_next = apply_clahe(vol[idx_next])

        img_25d = np.stack([slice_prev, slice_curr, slice_next], axis=-1)

        # Inférence
        masks, _, info = sam_inference(
            model,
            img_rgb=img_25d,
            urna_mask=urna_mask,
            mode="grid",
            make_colored=False
        )

        n_masks = len(masks)
        total_area = sum(m.sum() for m in masks) if n_masks > 0 else 0
        avg_area = total_area / n_masks if n_masks > 0 else 0

        results.append({
            'delta_z': delta_z,
            'n_masks': n_masks,
            'total_area': int(total_area),
            'avg_area': float(avg_area)
        })

        print(f"  → Masques détectés: {n_masks}")

    return pd.DataFrame(results)


def visualize_2d_vs_25d(comparison_results, output_dir):
    """Visualise comparaison 2D vs 2.5D"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Ligne 1: 2D
    axes[0, 0].imshow(comparison_results['2d']['img'])
    axes[0, 0].set_title("Image 2D")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(comparison_results['urna_mask'], cmap='gray')
    axes[0, 1].set_title("Masque urne")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(comparison_results['2d']['colored'])
    axes[0, 2].set_title(f"Segmentation 2D ({comparison_results['2d']['info']['n_masks_filtered']} masques)")
    axes[0, 2].axis('off')

    # Ligne 2: 2.5D
    axes[1, 0].imshow(comparison_results['25d']['img'])
    axes[1, 0].set_title("Image 2.5D")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(comparison_results['urna_mask'], cmap='gray')
    axes[1, 1].set_title("Masque urne")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(comparison_results['25d']['colored'])
    axes[1, 2].set_title(f"Segmentation 2.5D ({comparison_results['25d']['info']['n_masks_filtered']} masques)")
    axes[1, 2].axis('off')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path / "exp3_2d_vs_25d.png", dpi=150)
    print(f"\nGraphique sauvegardé: {output_path / 'exp3_2d_vs_25d.png'}")
    plt.show()


def visualize_delta_z_impact(df_delta_z, output_dir):
    """Visualise impact de delta_z"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df_delta_z['delta_z'], df_delta_z['n_masks'], 'o-', linewidth=2)
    ax1.set_xlabel('Delta_z')
    ax1.set_ylabel('Nombre de masques')
    ax1.set_title('Impact de delta_z sur le nombre de masques')
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_delta_z['delta_z'], df_delta_z['avg_area'], 's-', linewidth=2, color='orange')
    ax2.set_xlabel('Delta_z')
    ax2.set_ylabel('Aire moyenne (pixels)')
    ax2.set_title('Impact de delta_z sur l\'aire moyenne')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir)
    plt.savefig(output_path / "exp3_delta_z_impact.png", dpi=150)
    print(f"Graphique sauvegardé: {output_path / 'exp3_delta_z_impact.png'}")
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

    # Comparaison 2D vs 2.5D
    comparison = compare_2d_vs_25d(model, vol, mid)

    # Analyse IoU
    iou_25d = run_iou_analysis_25d(model, vol, mid, num_slices=10)

    # Test delta_z
    df_delta_z = test_delta_z_variations(model, vol, mid)

    # Visualisations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(Config.OUTPUT_DIR) / f"exp3_25d_{timestamp}"

    visualize_2d_vs_25d(comparison, output_dir)
    visualize_delta_z_impact(df_delta_z, output_dir)

    # Sauvegarder résultats
    np.savez(
        output_dir / "results.npz",
        iou_25d=iou_25d,
        mean_iou=np.mean(iou_25d),
        n_masks_2d=comparison['2d']['info']['n_masks_filtered'],
        n_masks_25d=comparison['25d']['info']['n_masks_filtered']
    )

    df_delta_z.to_csv(output_dir / "delta_z_results.csv", index=False)
    print(f"\nRésultats sauvegardés dans: {output_dir}")


if __name__ == "__main__":
    main()
