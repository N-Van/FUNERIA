"""
Fonctions d'inférence SAM avec différents modes
"""
import numpy as np
import torch
from utils import make_grid_points, colorize_masks
from config import Config


def sam_inference(
    model,
    img_rgb,
    urna_mask,
    mode="grid",
    grid_stride=None,
    points=None,
    labels=None,
    device=None,
    imgsz=None,
    iou=None,
    max_det=None,
    conf=None,
    min_area=None,
    max_area_ratio=None,
    make_colored=True,
    color_seed=0
):
    """
    Inférence SAM avec prompts (grid, points, ou auto)
    """
    # Paramètres par défaut
    if grid_stride is None:
        grid_stride = Config.GRID_STRIDE
    if device is None:
        device = Config.DEVICE
    if imgsz is None:
        imgsz = Config.IMGSZ
    if iou is None:
        iou = Config.IOU
    if max_det is None:
        max_det = Config.MAX_DET
    if conf is None:
        conf = Config.CONF_THR
    if min_area is None:
        min_area = Config.MIN_AREA
    if max_area_ratio is None:
        max_area_ratio = Config.MAX_AREA_RATIO
    
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img rgb doit être RGB [H,W,3] reçu {img_rgb.shape}")

    H, W, _ = img_rgb.shape
    img_gray = img_rgb[..., :3].mean(axis=2).astype(np.uint8)
    img_area = H * W
    max_area = int(max_area_ratio * img_area)

    if urna_mask is not None:
        urna_mask = urna_mask.astype(bool)
        if urna_mask.shape != (H, W):
            raise ValueError(f"urna_mask doit être de taille {(H,W)} reçu {urna_mask.shape}")

    # Préparation des prompts
    filtered_points = None
    filtered_labels = None

    if mode == "grid":
        pts = make_grid_points(H, W, grid_stride)

        if urna_mask is None:
            filtered_points = pts
        else:
            filtered_points = [(x, y) for (x, y) in pts if urna_mask[y, x]]

        filtered_labels = [1] * len(filtered_points)

        if len(filtered_points) == 0:
            colored = np.dstack([img_gray]*3).astype(np.uint8)
            return [], colored, {"mode": "grid", "n_masks_raw": 0,
                                "n_masks_filtered": 0, "n_points": 0}

    elif mode == "points":
        if points is None:
            raise ValueError("mode='points' mais aucun points fourni")
        filtered_points = points
        filtered_labels = labels if labels is not None else [1]*len(points)

    elif mode == "auto":
        filtered_points = None
        filtered_labels = None

    else:
        raise ValueError("mode doit être 'grid', 'auto' ou 'points'")

    # Inférence
    with torch.inference_mode():
        if filtered_points is not None:
            results = model(
                img_rgb,
                points=filtered_points,
                labels=filtered_labels,
                device=device,
                imgsz=imgsz,
                iou=iou,
                max_det=max_det,
                conf=conf,
                agnostic_nms=True,
                show=False,
            )
        else:
            results = model(
                img_rgb,
                device=device,
                imgsz=imgsz,
                iou=iou,
                max_det=max_det,
                conf=conf,
                agnostic_nms=True,
                show=False,
            )

    res = results[0]

    if res.masks is None:
        colored = np.dstack([img_gray] * 3).astype(np.uint8)
        return [], colored, {"mode": mode, "n_masks_raw": 0, "n_masks_filtered": 0, "n_points": 0}

    masks = res.masks.data.cpu().numpy().astype(np.uint8)
    masks = np.squeeze(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]

    print(f"[{mode}] masks brutes détectés", masks.shape)

    if urna_mask is None:
        print("[INFO] Pas de masque d'urne fourni : aucun filtrage appliqué.")
        filtered_masks = (masks > 0)
        colored = colorize_masks(img_gray, filtered_masks, seed=color_seed) if make_colored else None
        return list(filtered_masks), colored, {
            "mode": mode,
            "n_masks_raw": int(masks.shape[0]),
            "n_masks_filtered": int(filtered_masks.shape[0]),
            "n_points": 0 if filtered_points is None else len(filtered_points),
            "min_area": min_area,
            "max_area": max_area,
        }

    # Restreindre les masques à l'intérieur de l'urne
    masks = np.array([(m > 0) & urna_mask for m in masks], dtype=bool)

    # Filtrage par aire
    areas = np.array([m.sum() for m in masks])
    keep_idx = np.where((areas >= min_area) & (areas <= max_area))[0]
    filtered_masks = masks[keep_idx]

    print(f"[{mode}] masks filtrés", filtered_masks.shape)

    # Colorisation
    if make_colored:
        if filtered_masks.size == 0:
            colored = np.dstack([img_gray] * 3).astype(np.uint8)
        else:
            colored = colorize_masks(img_gray, filtered_masks, seed=color_seed)
    else:
        colored = None

    info = {
        "mode": mode,
        "n_masks_raw": int(masks.shape[0]),
        "n_masks_filtered": int(filtered_masks.shape[0]),
        "n_points": 0 if filtered_points is None else len(filtered_points),
        "min_area": min_area,
        "max_area": max_area,
    }

    return list(filtered_masks), colored, info


def sam_inference_boxes(
    model,
    img_rgb,
    urna_mask,
    boxes,
    device=None,
    imgsz=None,
    iou=None,
    max_det=None,
    conf=None,
    min_area=None,
    max_area_ratio=None,
    make_colored=True,
    color_seed=0
):
    """
    Inférence SAM avec prompts de type bounding boxes
    """
    # Paramètres par défaut
    if device is None:
        device = Config.DEVICE
    if imgsz is None:
        imgsz = Config.IMGSZ
    if iou is None:
        iou = Config.IOU
    if max_det is None:
        max_det = Config.MAX_DET
    if conf is None:
        conf = Config.CONF_THR
    if min_area is None:
        min_area = Config.MIN_AREA
    if max_area_ratio is None:
        max_area_ratio = Config.MAX_AREA_RATIO
    
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img rgb doit être RGB [H,W,3] reçu {img_rgb.shape}")

    H, W, _ = img_rgb.shape
    img_gray = img_rgb[..., :3].mean(axis=2).astype(np.uint8)
    img_area = H * W
    max_area = int(max_area_ratio * img_area)

    if urna_mask is not None:
        urna_mask = urna_mask.astype(bool)
        if urna_mask.shape != (H, W):
            raise ValueError(f"urna_mask doit être de taille {(H,W)} reçu {urna_mask.shape}")

    # Filtrer les boxes qui sont dans l'urne
    if urna_mask is not None:
        filtered_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if urna_mask[cy, cx]:
                filtered_boxes.append(box)
    else:
        filtered_boxes = boxes

    if len(filtered_boxes) == 0:
        colored = np.dstack([img_gray] * 3).astype(np.uint8)
        return [], colored, {"mode": "boxes", "n_masks_raw": 0,
                            "n_masks_filtered": 0, "n_boxes": 0}

    with torch.inference_mode():
        results = model(
            img_rgb,
            bboxes=filtered_boxes,
            device=device,
            imgsz=imgsz,
            iou=iou,
            max_det=max_det,
            conf=conf,
            agnostic_nms=True,
            show=False,
        )

    res = results[0]

    if res.masks is None:
        colored = np.dstack([img_gray] * 3).astype(np.uint8)
        return [], colored, {"mode": "boxes", "n_masks_raw": 0, "n_masks_filtered": 0, "n_boxes": len(filtered_boxes)}

    masks = res.masks.data.cpu().numpy().astype(np.uint8)
    masks = np.squeeze(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]

    print(f"[boxes] masks brutes détectés", masks.shape)

    if urna_mask is None:
        filtered_masks = (masks > 0)
        colored = colorize_masks(img_gray, filtered_masks, seed=color_seed) if make_colored else None
        return list(filtered_masks), colored, {
            "mode": "boxes",
            "n_masks_raw": int(masks.shape[0]),
            "n_masks_filtered": int(filtered_masks.shape[0]),
            "n_boxes": len(filtered_boxes),
            "min_area": min_area,
            "max_area": max_area,
        }

    # Restreindre et filtrer
    masks = np.array([(m > 0) & urna_mask for m in masks], dtype=bool)
    areas = np.array([m.sum() for m in masks])
    keep_idx = np.where((areas >= min_area) & (areas <= max_area))[0]
    filtered_masks = masks[keep_idx]

    print(f"[boxes] masks filtrés", filtered_masks.shape)

    if make_colored:
        if filtered_masks.size == 0:
            colored = np.dstack([img_gray] * 3).astype(np.uint8)
        else:
            colored = colorize_masks(img_gray, filtered_masks, seed=color_seed)
    else:
        colored = None

    info = {
        "mode": "boxes",
        "n_masks_raw": int(masks.shape[0]),
        "n_masks_filtered": int(filtered_masks.shape[0]),
        "n_boxes": len(filtered_boxes),
        "min_area": min_area,
        "max_area": max_area,
    }

    return list(filtered_masks), colored, info
