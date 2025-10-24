from codecarbon import EmissionsTracker
import tifffile
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

# Initialiser le tracker
tracker = EmissionsTracker()
tracker.start()

# Charger l'image 3D
image_3d = tifffile.imread("tests/Romane_Martin_urne_sature_10-4.tif")
print(f"Shape de l'image 3D: {image_3d.shape}")
print(f"Type de données: {image_3d.dtype}")

middle_idx = len(image_3d) // 2
print(f"\nTraitement de la slice du milieu: {middle_idx}/{len(image_3d)}")

slice_img = image_3d[middle_idx]

if len(slice_img.shape) == 2:
    if slice_img.dtype != np.uint8:
        slice_min = slice_img.min()
        slice_max = slice_img.max()
        if slice_max > slice_min:
            slice_normalized = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            slice_normalized = np.zeros_like(slice_img, dtype=np.uint8)
    else:
        slice_normalized = slice_img
    
    slice_img_rgb = np.stack([slice_normalized] * 3, axis=-1)
else:
    if slice_img.dtype != np.uint8:
        slice_min = slice_img.min()
        slice_max = slice_img.max()
        if slice_max > slice_min:
            slice_img_rgb = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            slice_img_rgb = np.zeros_like(slice_img, dtype=np.uint8)
    else:
        slice_img_rgb = slice_img

cv2.imwrite(f"slice_{middle_idx}_original.png", slice_img_rgb)
print(f"✓ Image originale sauvegardée: slice_{middle_idx}_original.png")

print("\n" + "="*60)
print("SEGMENTATION AVEC SAM")
print("="*60)

print("\nChargement du modèle SAM...")
sam_checkpoint = "tests/sam_vit_b_01ec64.pth" 
model_type = "vit_b"  

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Créer le générateur de masques automatique
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=50,
    pred_iou_thresh=0.90,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,  
)

print("Génération des masques avec SAM...")
masks = mask_generator.generate(slice_img_rgb)

print(f"✓ SAM: {len(masks)} objet(s) détecté(s)")

if len(masks) > 0:
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    segmentation_colored = slice_img_rgb.copy()
    segmentation_map = np.zeros(slice_img_rgb.shape[:2], dtype=np.uint8)
    
    np.random.seed(42)  
    colors = []
    for i in range(len(masks_sorted)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)
    
    print(f"\nInformations sur les objets détectés:")
    print(f"{'ID':<5} {'Aire (pixels)':<15} {'Stabilité':<12} {'Couleur RGB'}")
    print("-" * 60)
    
    for j, mask_dict in enumerate(masks_sorted):
        mask = mask_dict['segmentation']
        area = mask_dict['area']
        stability_score = mask_dict['stability_score']
        
        segmentation_map[mask] = j + 1
        
        color = colors[j]
        overlay = slice_img_rgb.copy()
        overlay[mask] = color
        
        alpha = 0.5
        segmentation_colored = cv2.addWeighted(segmentation_colored, 1, overlay, alpha, 0)
        
        print(f"{j:<5} {area:<15} {stability_score:<12.3f} {color}")
    
    cv2.imwrite(f"sam_segmentation_colored_slice_{middle_idx}.png", segmentation_colored)
    print(f"\n✓ Image de segmentation colorée sauvegardée: sam_segmentation_colored_slice_{middle_idx}.png")
    
    segmentation_colored_map = cv2.applyColorMap((segmentation_map * 255 // len(masks_sorted)).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"sam_segmentation_map_slice_{middle_idx}.png", segmentation_colored_map)
    print(f"✓ Carte de segmentation sauvegardée: sam_segmentation_map_slice_{middle_idx}.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image originale
    axes[0].imshow(slice_img_rgb)
    axes[0].set_title('Image Originale', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation colorée
    axes[1].imshow(segmentation_colored)
    axes[1].set_title(f'Segmentation SAM ({len(masks_sorted)} objets)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Carte de segmentation
    axes[2].imshow(segmentation_map, cmap='nipy_spectral')
    axes[2].set_title('Carte de Segmentation', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"sam_visualization_slice_{middle_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualisation comparative sauvegardée: sam_visualization_slice_{middle_idx}.png")
    
else:
    print("\n✗ Aucun objet détecté par SAM")
    print("\nℹ️  Essayez de:")
    print("  - Réduire min_mask_region_area (objets plus petits)")
    print("  - Augmenter points_per_side (plus de détails)")
    print("  - Réduire pred_iou_thresh et stability_score_thresh (moins strict)")

# Arrêter le suivi
tracker.stop()

print("\n" + "="*60)
print("TRAITEMENT TERMINÉ")
print("="*60)
print(f"Slice traitée: {middle_idx}")
print(f"\nFichiers générés:")
print(f"  - slice_{middle_idx}_original.png")
print(f"  - sam_segmentation_colored_slice_{middle_idx}.png (VOTRE IMAGE COLORÉE)")
print(f"  - sam_segmentation_map_slice_{middle_idx}.png")
print(f"  - sam_visualization_slice_{middle_idx}.png")