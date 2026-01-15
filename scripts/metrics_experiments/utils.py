"""
Fonctions utilitaires pour le prétraitement et la visualisation
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import Config

def get_urna_mask_threshold(img_gray, thr=None):
    """Détection de l'urne par seuillage"""
    if thr is None:
        thr = Config.URNA_THRESHOLD
    
    if img_gray.ndim != 2:
        raise ValueError(f"get_urna_mask_threshold attend une image 2D grayscale, reçu shape={img_gray.shape}")
    
    urna_mask = img_gray >= thr
    return urna_mask


def apply_clahe(sl, use_clahe=True):
    """Applique CLAHE pour améliorer le contraste"""
    sl_prep = sl.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMIT, 
            tileGridSize=Config.CLAHE_TILE_GRID_SIZE
        )
        sl_prep = clahe.apply(sl_prep)
    return sl_prep


def convert_rgb(picture_3D: np.ndarray) -> np.ndarray:
    """Convertit [H,W] en [H,W,3]"""
    return np.repeat(picture_3D[..., np.newaxis], 3, -1)


def normalize_to_uint8(img):
    """Normalise une image en uint8"""
    if img.dtype == np.uint8:
        return img
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def colorize_masks(image, masks_bool, seed=42):
    """Colorisation des masques pour affichage"""
    img = np.asarray(image)
    if img.ndim == 2:
        out = np.dstack([img, img, img]).astype(np.uint8, copy=False)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = img.astype(np.uint8, copy=False).copy()
    else:
        raise ValueError(f"image must be [H,W] or [H,W,3], got {img.shape}")
    
    H, W = out.shape[:2]
    rng = np.random.default_rng(seed)
    
    for m in masks_bool:
        m = np.asarray(m, dtype=bool)
        if m.shape != (H, W):
            raise ValueError(f"mask shape {m.shape} != {(H,W)}")
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        out[m] = color
    
    return out


def make_grid_points(h, w, stride, label=1):
    """Génération de points en grille"""
    xs = np.arange(stride // 2, w, stride)
    ys = np.arange(stride // 2, h, stride)
    pts = [(int(x), int(y)) for y in ys for x in xs]
    return pts


def make_grid_boxes(h, w, stride, box_size=64):
    """Génère une grille de bounding boxes"""
    xs = np.arange(stride // 2, w, stride)
    ys = np.arange(stride // 2, h, stride)
    boxes = []
    for y in ys:
        for x in xs:
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            x2 = min(w, x + box_size // 2)
            y2 = min(h, y + box_size // 2)
            boxes.append([x1, y1, x2, y2])
    return boxes


def match_masks_between_slices(masks_prev, masks_curr, iou_threshold=0.3):
    """Calcule l'IoU entre deux ensembles de masques combinés"""
    if len(masks_prev) == 0 or len(masks_curr) == 0:
        return 0.0

    # Combiner les masques dans un seul mask binaire
    mask_prev_combined = np.zeros_like(masks_prev[0], dtype=bool)
    for mask in masks_prev:
        mask_prev_combined |= mask

    mask_curr_combined = np.zeros_like(masks_curr[0], dtype=bool)
    for mask in masks_curr:
        mask_curr_combined |= mask

    # Calculer l'IoU
    intersection = np.logical_and(mask_prev_combined, mask_curr_combined)
    union = np.logical_or(mask_prev_combined, mask_curr_combined)

    if np.sum(union) == 0:
        return 0.0

    iou = np.sum(intersection) / np.sum(union)
    return iou


def plot_comparison(images_dict, titles_dict, figsize=(16, 12), suptitle="Comparison"):
    """
    Affiche plusieurs images côte à côte
    
    Args:
        images_dict: dict {key: image}
        titles_dict: dict {key: title}
        figsize: taille de la figure
        suptitle: titre global
    """
    n_images = len(images_dict)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (key, img) in zip(axes, images_dict.items()):
        ax.imshow(img)
        ax.set_title(titles_dict.get(key, key))
        ax.axis('off')
    
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()
