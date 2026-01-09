import numpy as np
import cv2


def normalize_to_uint8(img):
    """Normalise une image en uint8"""
    if img.dtype == np.uint8:
        return img
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def apply_clahe(sl, use_clahe=True):
    sl_prep=sl.copy()
    if use_clahe:
        sl_prep = clahe.apply(sl_prep)
    return sl_prep  # [H,W]

def convert_rgb(picture_3D: np.ndarray) -> np.ndarray:
    # convert [H,W] en  [H,W,3]
    return np.repeat(picture_3D[..., np.newaxis], 3, -1)

def create_25d_image(volume, slice_idx, normalize=True):
    D, H, W = volume.shape

    # Gérer les cas limites (bords du volume)
    idx_prev = max(0, slice_idx - 1)
    idx_curr = slice_idx
    idx_next = min(D - 1, slice_idx + 1)

    # Extraire les slices
    slice_prev = volume[idx_prev]
    slice_curr = volume[idx_curr]
    slice_next = volume[idx_next]

    # Normaliser chaque slice indépendamment si demandé
    if normalize:
        slice_prev = normalize_to_uint8(slice_prev)
        slice_curr = normalize_to_uint8(slice_curr)
        slice_next = normalize_to_uint8(slice_next)

    img_25d = np.stack([slice_prev, slice_curr, slice_next], axis=-1)

    return img_25d


def create_25d_image_with_clahe(volume, slice_idx, use_clahe=True):
    """
    Version avec CLAHE appliqué à chaque slice avant empilement
    """
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
