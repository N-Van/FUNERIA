#!/usr/bin/env python3
"""
Analyse de l'autocorrélation du dataset avec plusieurs métriques.
Ce script calcule différentes métriques (SSIM, RSE, MAE, NCC, etc.) 
pour différents deltas et affiche les résultats dans une seule figure.

Usage:
  python analyze_dataset_metrics.py --tiff path/to/stack.tif --deltas 1 2 3 5 10 20
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import tifffile
except ImportError:
    tifffile = None

from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


def load_stack(path: str) -> np.ndarray:
    """Charge un stack TIFF multi-pages."""
    if tifffile is not None:
        arr = tifffile.imread(path)
    else:
        import imageio
        arr = imageio.mimread(path)
        arr = np.stack(arr, axis=0)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr)


# ============== MÉTRIQUES ==============

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index (SSIM)."""
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    dr = max(a.max() - a.min(), b.max() - b.min(), 1.0)
    return structural_similarity(a, b, data_range=dr)


def compute_rse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Relative Squared Error (RSE) = ||img1 - img2||^2 / ||img1||^2."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    diff_norm = np.sum((a - b) ** 2)
    ref_norm = np.sum(a ** 2)
    if ref_norm == 0:
        return 0.0 if diff_norm == 0 else np.inf
    return diff_norm / ref_norm


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return np.mean(np.abs(a - b))


def compute_nrmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized Root Mean Squared Error (NRMSE)."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    rmse = np.sqrt(mse)
    # Normalisation par la plage de l'image de référence
    data_range = a.max() - a.min()
    if data_range == 0:
        return 0.0 if rmse == 0 else np.inf
    return rmse / data_range


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (PSNR)."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    dr = max(a.max() - a.min(), b.max() - b.min(), 1.0)
    # Éviter PSNR infini si images identiques
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0  # Valeur cap pour images identiques
    return peak_signal_noise_ratio(a, b, data_range=dr)


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized Cross-Correlation (NCC)."""
    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64).flatten()
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return np.dot(a_centered, b_centered) / (norm_a * norm_b)


def compute_pearson(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pearson Correlation Coefficient."""
    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64).flatten()
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return np.corrcoef(a, b)[0, 1]


# Dictionnaire des métriques disponibles
METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "SSIM": compute_ssim,
    "RSE": compute_rse,
    "MAE": compute_mae,
    "NRMSE": compute_nrmse,
    "PSNR": compute_psnr,
    "NCC": compute_ncc,
    "Pearson": compute_pearson,
}


def compute_metric_for_delta(stack: np.ndarray, delta: int, 
                              metric_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule une métrique pour des paires (i, i+delta) avec pas de 1.
    Retourne (scores, indices).
    """
    n = stack.shape[0]
    if delta < 1 or n <= delta:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    
    scores = np.zeros(n - delta, dtype=float)
    indices = np.arange(0, n - delta, dtype=int)
    
    for idx, i in enumerate(indices):
        scores[idx] = metric_func(stack[i], stack[i + delta])
    
    return scores, indices


def compute_all_metrics_for_deltas(stack: np.ndarray, deltas: List[int], 
                                    metric_names: List[str]) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Calcule les statistiques (mean, std) de chaque métrique pour chaque delta.
    
    Retourne un dictionnaire:
        {metric_name: {delta: (mean, std), ...}, ...}
    """
    results = {name: {} for name in metric_names}
    
    for delta in deltas:
        print(f"  Computing delta={delta}...")
        for name in metric_names:
            scores, _ = compute_metric_for_delta(stack, delta, METRICS[name])
            if scores.size > 0:
                results[name][delta] = (np.mean(scores), np.std(scores))
            else:
                results[name][delta] = (np.nan, np.nan)
    
    return results


def plot_autocorrelation_analysis(results: Dict[str, Dict[int, Tuple[float, float]]], 
                                   deltas: List[int],
                                   outdir: str,
                                   filename: str = "autocorrelation_analysis.png"):
    """
    Crée une figure unique avec toutes les métriques en fonction des deltas.
    """
    metric_names = list(results.keys())
    n_metrics = len(metric_names)
    
    # Définir les couleurs pour chaque métrique
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
    
    # Créer la figure avec plusieurs sous-graphiques
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # ====== Subplot 1: Toutes les métriques normalisées ======
    ax1 = fig.add_subplot(gs[0, :])
    
    for idx, name in enumerate(metric_names):
        means = [results[name].get(d, (np.nan, np.nan))[0] for d in deltas]
        stds = [results[name].get(d, (np.nan, np.nan))[1] for d in deltas]
        
        # Normaliser entre 0 et 1 pour comparaison
        means_arr = np.array(means)
        valid_mask = ~np.isnan(means_arr) & ~np.isinf(means_arr)
        if valid_mask.any():
            min_val = np.nanmin(means_arr[valid_mask])
            max_val = np.nanmax(means_arr[valid_mask])
            if max_val > min_val:
                means_norm = (means_arr - min_val) / (max_val - min_val)
            else:
                means_norm = np.zeros_like(means_arr)
        else:
            means_norm = np.zeros_like(means_arr)
        
        ax1.plot(deltas, means_norm, 'o-', color=colors[idx], label=name, linewidth=2, markersize=6)
    
    ax1.set_xlabel("Delta (distance entre slices)", fontsize=11)
    ax1.set_ylabel("Valeur normalisée [0-1]", fontsize=11)
    ax1.set_title("Évolution des métriques normalisées en fonction du delta", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(deltas)
    
    # ====== Subplots individuels pour chaque métrique ======
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    
    # Grouper les métriques par type pour meilleure visualisation
    similarity_metrics = ["SSIM", "NCC", "Pearson"]  # Plus haut = plus similaire
    error_metrics = ["RSE", "MAE", "NRMSE"]  # Plus bas = plus similaire
    special_metrics = ["PSNR"]  # Plus haut = meilleur
    
    # Subplot 2: Métriques de similarité
    ax2 = fig.add_subplot(gs[1, 0])
    for name in similarity_metrics:
        if name in results:
            means = [results[name].get(d, (np.nan, np.nan))[0] for d in deltas]
            stds = [results[name].get(d, (np.nan, np.nan))[1] for d in deltas]
            color = colors[metric_names.index(name)]
            ax2.errorbar(deltas, means, yerr=stds, fmt='o-', color=color, 
                        label=name, linewidth=2, markersize=5, capsize=3)
    
    ax2.set_xlabel("Delta", fontsize=10)
    ax2.set_ylabel("Valeur", fontsize=10)
    ax2.set_title("Métriques de similarité\n(plus haut = plus similaire)", fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(deltas)
    
    # Subplot 3: Métriques d'erreur
    ax3 = fig.add_subplot(gs[1, 1])
    for name in error_metrics:
        if name in results:
            means = [results[name].get(d, (np.nan, np.nan))[0] for d in deltas]
            stds = [results[name].get(d, (np.nan, np.nan))[1] for d in deltas]
            color = colors[metric_names.index(name)]
            ax3.errorbar(deltas, means, yerr=stds, fmt='o-', color=color, 
                        label=name, linewidth=2, markersize=5, capsize=3)
    
    ax3.set_xlabel("Delta", fontsize=10)
    ax3.set_ylabel("Valeur", fontsize=10)
    ax3.set_title("Métriques d'erreur\n(plus bas = plus similaire)", fontsize=11)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(deltas)
    
    # Subplot 4: PSNR
    ax4 = fig.add_subplot(gs[2, 0])
    if "PSNR" in results:
        means = [results["PSNR"].get(d, (np.nan, np.nan))[0] for d in deltas]
        stds = [results["PSNR"].get(d, (np.nan, np.nan))[1] for d in deltas]
        color = colors[metric_names.index("PSNR")]
        ax4.errorbar(deltas, means, yerr=stds, fmt='o-', color=color, 
                    label="PSNR", linewidth=2, markersize=5, capsize=3)
    
    ax4.set_xlabel("Delta", fontsize=10)
    ax4.set_ylabel("PSNR (dB)", fontsize=10)
    ax4.set_title("Peak Signal-to-Noise Ratio\n(plus haut = meilleur)", fontsize=11)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(deltas)
    
    # Subplot 5: Tableau récapitulatif
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Créer le tableau
    table_data = [["Delta"] + [str(d) for d in deltas]]
    for name in metric_names:
        row = [name]
        for d in deltas:
            mean, std = results[name].get(d, (np.nan, np.nan))
            if np.isnan(mean):
                row.append("N/A")
            else:
                row.append(f"{mean:.4f}")
        table_data.append(row)
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Colorer l'en-tête
    for j in range(len(deltas) + 1):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(metric_names) + 1):
        table[(i, 0)].set_facecolor('#D9E2F3')
        table[(i, 0)].set_text_props(fontweight='bold')
    
    ax5.set_title("Tableau récapitulatif des valeurs moyennes", fontsize=11, pad=20)
    
    # Titre principal
    fig.suptitle("Analyse d'autocorrélation du dataset\nVariation des métriques en fonction de la distance inter-slices", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Sauvegarder
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure sauvegardée: {filepath}")
    
    return filepath


def save_results_to_csv(results: Dict[str, Dict[int, Tuple[float, float]]], 
                        deltas: List[int], outdir: str, 
                        filename: str = "autocorrelation_metrics.csv"):
    """Sauvegarde les résultats dans un fichier CSV."""
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)
    
    with open(filepath, 'w') as f:
        # En-tête
        f.write("Metric,Delta,Mean,Std\n")
        for name in results:
            for d in deltas:
                mean, std = results[name].get(d, (np.nan, np.nan))
                f.write(f"{name},{d},{mean:.6f},{std:.6f}\n")
    
    print(f"Résultats CSV sauvegardés: {filepath}")
    return filepath


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyse d'autocorrélation avec plusieurs métriques sur un stack TIFF"
    )
    parser.add_argument("--tiff", type=str, required=True, 
                        help="Chemin vers le fichier TIFF multi-pages")
    parser.add_argument("--deltas", type=int, nargs='+', default=[1, 2, 3, 5, 10, 20, 50],
                        help="Liste des deltas à analyser (défaut: 1 2 3 5 10 20 50)")
    parser.add_argument("--metrics", type=str, nargs='+', 
                        default=["SSIM", "RSE", "MAE", "NRMSE", "PSNR", "NCC", "Pearson"],
                        help="Liste des métriques à calculer")
    parser.add_argument("--outdir", type=str, default="ssim_results",
                        help="Dossier de sortie pour les résultats")
    parser.add_argument("--start", type=int, default=None,
                        help="Index de début (inclusif)")
    parser.add_argument("--end", type=int, default=None,
                        help="Index de fin (exclusif)")
    parser.add_argument("--margin", type=int, default=80,
                        help="Si start/end non fournis, tronquer cette valeur au début et à la fin (défaut: 80)")
    
    args = parser.parse_args(argv)
    
    # Vérifier le fichier TIFF
    if not os.path.exists(args.tiff):
        print(f"ERREUR: Fichier TIFF non trouvé: {args.tiff}")
        return 1
    
    # Vérifier les métriques demandées
    for m in args.metrics:
        if m not in METRICS:
            print(f"ERREUR: Métrique inconnue '{m}'. Disponibles: {list(METRICS.keys())}")
            return 1
    
    # Charger le stack
    print(f"Chargement du TIFF: {args.tiff}")
    stack = load_stack(args.tiff)
    
    # Appliquer la plage si spécifiée. Sinon, appliquer margin trimming si possible
    if args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end or stack.shape[0]
        stack = stack[s:e]
    else:
        margin = int(args.margin or 0)
        if margin > 0 and stack.shape[0] > 2 * margin:
            s = margin
            e = stack.shape[0] - margin
            print(f"Applying margin trim: using slices [{s}:{e}] (margin={margin})")
            stack = stack[s:e]
    
    print(f"Shape du stack: {stack.shape}")
    print(f"Deltas à analyser: {args.deltas}")
    print(f"Métriques: {args.metrics}")
    
    # Filtrer les deltas valides
    valid_deltas = [d for d in args.deltas if d < stack.shape[0]]
    if len(valid_deltas) < len(args.deltas):
        print(f"Note: Certains deltas ignorés (> nombre de slices)")
    
    # Calculer toutes les métriques
    print("\nCalcul des métriques...")
    results = compute_all_metrics_for_deltas(stack, valid_deltas, args.metrics)
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ANALYSE D'AUTOCORRÉLATION")
    print("="*60)
    
    for name in args.metrics:
        print(f"\n{name}:")
        for d in valid_deltas:
            mean, std = results[name].get(d, (np.nan, np.nan))
            print(f"  delta={d:3d}: mean={mean:.4f}, std={std:.4f}")
    
    # Créer la figure combinée
    plot_autocorrelation_analysis(results, valid_deltas, args.outdir)
    
    # Sauvegarder en CSV
    save_results_to_csv(results, valid_deltas, args.outdir)
    
    # Sauvegarder en NPY
    np.save(os.path.join(args.outdir, "autocorrelation_results.npy"), results)
    print(f"Résultats NPY sauvegardés: {os.path.join(args.outdir, 'autocorrelation_results.npy')}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
