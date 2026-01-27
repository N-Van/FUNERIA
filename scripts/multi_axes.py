#!/usr/bin/env python3
"""Multi-axis 3D segmentation with SAM and ensemble fusion.

Performs segmentation along all three axes (axial, sagittal, coronal) using SAM
with prompt points, then combines them for coherent 3D segmentation.

this script is not to run: it has some memory crash issues, but it is kept for ideas reference.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tifffile
from tqdm import tqdm

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using alternative implementations")


def load_volume(path: str) -> np.ndarray:
    """Load 3D volume from TIFF file."""
    vol = tifffile.imread(path)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    return vol


def normalize_volume(vol: np.ndarray, percentile: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """Normalize volume to [0, 1] range using percentile clipping."""
    vol = vol.astype(np.float32)
    plow, phigh = np.percentile(vol, percentile)
    vol = np.clip(vol, plow, phigh)
    vol = (vol - plow) / (phigh - plow + 1e-8)
    return vol


def prepare_frame_for_sam(slice_2d: np.ndarray) -> torch.Tensor:
    """Convert grayscale slice to RGB format expected by SAM (1, 3, H, W)."""
    # Normalize to [0, 1] 
    if slice_2d.max() > 1.0:
        slice_2d = slice_2d / 255.0
    
    # Convert to RGB by repeating channels
    rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=0)  # (3, H, W)
    
    # Add batch dimension
    frame = torch.from_numpy(rgb).float().unsqueeze(0)  # (1, 3, H, W)
    
    return frame


def load_sam_module(model_path: str, device: str = "cuda"):
    from src.models.sam_module import SAM3DModuleLinear
    
    print(f"Loading SAM checkpoint from: {model_path}")
    model = SAM3DModuleLinear.load_from_checkpoint(
        model_path,
        map_location=device
    )
    model = model.to(device)
    model.eval()
    
    return model


def segment_axis_with_sam(
    volume: np.ndarray,
    sam_module: torch.nn.Module,
    axis: int,
    device: str = "cuda",
    grid_stride: int = 32,
    points_batch_size: int = 25,
    mode: str = "grid",
    imgsz: Optional[int] = None,
    clear_cache_every: int = 10,
) -> np.ndarray:
    """
    Segment volume along a specific axis using SAM.
    
    Args:
        volume: Input 3D volume (Z, Y, X), normalized to [0, 1]
        sam_module: SAM3DModuleLinear instance
        axis: Axis along which to iterate (0=Z, 1=Y, 2=X)
        device: Device for inference
        grid_stride: Stride for grid-based prompts
        points_batch_size: Batch size for processing points
        mode: SAM mode ("grid", "auto")
        imgsz: Image size for SAM
        clear_cache_every: Clear GPU cache every N slices
    
    Returns:
        Binary segmentation mask with same shape as volume
    """
    seg_map = np.zeros_like(volume, dtype=bool)
    n_slices = volume.shape[axis]
    
    axis_names = {0: "axial", 1: "sagittal", 2: "coronal"}
    print(f"Segmenting {n_slices} slices along {axis_names[axis]} axis (axis={axis})")
    
    with torch.no_grad():
        for i in tqdm(range(n_slices), desc=f"Axis {axis}"):
            # Extract slice
            if axis == 0:
                slice_2d = volume[i, :, :]
            elif axis == 1:
                slice_2d = volume[:, i, :]
            else:  # axis == 2
                slice_2d = volume[:, :, i]
            
            # Prepare frame for SAM (1, 3, H, W)
            frame = prepare_frame_for_sam(slice_2d).to(device)
            
            # Run SAM inference on single slice
            try:
                masks_f, info = sam_module.infer_one_projection(
                    frame,
                    urna_mask=None,
                    mode=mode,
                    grid_stride=grid_stride,
                    points_batch_size=points_batch_size,
                    imgsz=imgsz,
                )
                
                # Combine all masks for this slice
                if masks_f.shape[0] > 0:
                    slice_mask = masks_f.any(axis=0)  # Union of all masks
                else:
                    slice_mask = np.zeros_like(slice_2d, dtype=bool)
                
                # Store result
                if axis == 0:
                    seg_map[i, :, :] = slice_mask
                elif axis == 1:
                    seg_map[:, i, :] = slice_mask
                else:
                    seg_map[:, :, i] = slice_mask
                
            except RuntimeError as e:
                print(f"\nWarning: Failed to process slice {i}: {e}")
                print("Skipping this slice...")
                continue
            
            # Clear GPU cache periodically
            if device == "cuda" and (i + 1) % clear_cache_every == 0:
                torch.cuda.empty_cache()
    
    return seg_map


def compute_confidence_map(binary_map: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Compute confidence map based on local density of segmentation.
    Higher values indicate more confident regions.
    """
    if HAS_SCIPY:
        from scipy.ndimage import uniform_filter
        confidence = uniform_filter(binary_map.astype(float), size=kernel_size)
    else:
        # Use simple convolution as alternative
        from scipy.signal import convolve
        kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)
        confidence = convolve(binary_map.astype(float), kernel, mode='same')
    
    return confidence


def fuse_majority(seg_maps: Dict[int, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Fuse segmentations using majority voting."""
    vote_sum = sum(seg_maps.values())
    n_axes = len(seg_maps)
    
    # Majority vote (at least 2 out of 3 axes must agree)
    fused = (vote_sum >= (n_axes / 2 + 0.5)).astype(np.uint8)
    return fused


def fuse_weighted_average(
    seg_maps: Dict[int, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Fuse segmentations using weighted average."""
    if weights is None:
        # Equal weights
        weights = {axis: 1.0 / len(seg_maps) for axis in seg_maps.keys()}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Weighted sum
    fused_float = sum(seg_maps[axis].astype(float) * weights[axis] for axis in seg_maps.keys())
    fused = (fused_float > 0.5).astype(np.uint8)
    return fused


def fuse_uncertainty_based(seg_maps: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Fuse segmentations based on local confidence.
    Regions with higher local density get more weight.
    """
    # Compute confidence for each axis
    confidences = {}
    for axis, seg in seg_maps.items():
        confidences[axis] = compute_confidence_map(seg, kernel_size=5)
    
    # Normalize confidences to weights at each voxel
    total_confidence = sum(confidences.values()) + 1e-8
    
    # Weighted fusion
    fused_float = sum(
        seg_maps[axis].astype(float) * (confidences[axis] / total_confidence)
        for axis in seg_maps.keys()
    )
    
    fused = (fused_float > 0.5).astype(np.uint8)
    return fused


def fuse_consensus(
    seg_maps: Dict[int, np.ndarray],
    min_agreement: int = 2,
) -> np.ndarray:
    """Keep only regions where at least min_agreement axes agree."""
    vote_sum = sum(seg_maps.values())
    
    # Consensus: at least min_agreement axes must agree
    consensus = (vote_sum >= min_agreement).astype(np.uint8)
    
    return consensus


def postprocess_segmentation(
    seg: np.ndarray,
    min_size: int = 100,
    fill_holes: bool = True,
) -> np.ndarray:
    """Post-process segmentation to remove small objects and fill holes."""
    binary_seg = seg.astype(bool)
    
    if HAS_SCIPY:
        if min_size > 0:
            labeled, num_features = ndimage.label(binary_seg)
            sizes = ndimage.sum(binary_seg, labeled, range(num_features + 1))
            mask_sizes = sizes < min_size
            remove_small = mask_sizes[labeled]
            binary_seg[remove_small] = 0
        
        if fill_holes:
            binary_seg = ndimage.binary_fill_holes(binary_seg).astype(bool)
    else:
        try:
            from skimage import morphology, measure
            
            if min_size > 0:
                binary_seg = morphology.remove_small_objects(binary_seg, min_size=min_size)
            
            if fill_holes:
                binary_seg = morphology.remove_small_holes(binary_seg, area_threshold=min_size)
        except ImportError:
            print("Warning: Neither scipy nor skimage available, skipping postprocessing")
    
    return binary_seg.astype(np.uint8)


def compute_consistency_score(seg_maps: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Compute consistency score across axes.
    High score = all axes agree, low score = axes disagree.
    """
    # Variance across predictions
    segs_stack = np.stack([seg.astype(float) for seg in seg_maps.values()], axis=0)
    variance = np.var(segs_stack, axis=0)
    
    # Convert variance to consistency score (0=inconsistent, 1=consistent)
    # For binary data, variance is maximized at 0.25 (when p=0.5)
    consistency = 1 - np.clip(variance * 4, 0, 1)
    return consistency


def compute_dice_coefficient(seg1: np.ndarray, seg2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary segmentations."""
    intersection = np.logical_and(seg1, seg2).sum()
    union = seg1.sum() + seg2.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2 * intersection / union
    return float(dice)


def print_segmentation_stats(seg_maps: Dict[int, np.ndarray], axis_names: Dict[int, str]) -> None:
    print("\n=== Segmentation Statistics ===")
    
    total_voxels = seg_maps[0].size
    
    for axis, seg in seg_maps.items():
        n_positive = seg.sum()
        percentage = 100 * n_positive / total_voxels
        print(f"{axis_names[axis]:>10} axis: {n_positive:>10,} voxels ({percentage:>5.2f}%)")
    
    # Pairwise Dice scores
    print("\n=== Pairwise Dice Coefficients ===")
    axes = list(seg_maps.keys())
    for i, ax1 in enumerate(axes):
        for ax2 in axes[i+1:]:
            dice = compute_dice_coefficient(seg_maps[ax1], seg_maps[ax2])
            print(f"{axis_names[ax1]:>10} vs {axis_names[ax2]:>10}: {dice:.4f}")


def main(argv: Optional[list[str]] = None) -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Multi-axis 3D segmentation with SAM and ensemble fusion"
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input 3D volume (TIFF)")
    parser.add_argument("--output", "-o", required=True, help="Output segmentation (TIFF)")
    parser.add_argument("--model-path", required=True, help="Path to SAM checkpoint (.ckpt)")
    parser.add_argument(
        "--fusion-mode",
        choices=["majority", "weighted", "uncertainty", "consensus"],
        default="majority",
        help="Fusion strategy",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=32,
        help="Grid stride for SAM prompts (smaller = more points = slower but better)",
    )
    parser.add_argument(
        "--points-batch-size",
        type=int,
        default=25,
        help="Batch size for processing prompt points",
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "auto"],
        default="grid",
        help="SAM inference mode",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size for SAM (default: min(H,W))",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Minimum object size for postprocessing",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip postprocessing",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual axis segmentations",
    )
    parser.add_argument(
        "--save-consistency",
        action="store_true",
        help="Save consistency score map",
    )
    parser.add_argument(
        "--axes",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Axes to segment (0=Z, 1=Y, 2=X). Default: all three",
    )
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=10,
        help="Clear GPU cache every N slices to prevent OOM",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Data type for processing (float16 uses less memory)",
    )
    
    args = parser.parse_args(argv)
    
    # Validate axes
    for ax in args.axes:
        if ax not in [0, 1, 2]:
            print(f"ERROR: Invalid axis {ax}. Must be 0, 1, or 2.")
            return 1
    
    # Load volume
    print(f"Loading volume: {args.input}")
    volume = load_volume(args.input)
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Volume size: {volume.nbytes / (1024**3):.2f} GB")
    
    # Check memory before proceeding
    import psutil
    mem = psutil.virtual_memory()
    print(f"\nSystem memory: {mem.total / (1024**3):.1f} GB total, "
          f"{mem.available / (1024**3):.1f} GB available")
    
    if volume.nbytes * 4 > mem.available:  # Need ~4x for processing
        print("\nWARNING: May not have enough RAM. Consider:")
        print("  1. Using fewer axes (--axes 0 or --axes 0 2)")
        print("  2. Cropping the volume first")
        print("  3. Using --dtype float16")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Normalize to [0, 1]
    print("\nNormalizing volume...")
    volume = normalize_volume(volume)
    
    # Convert to specified dtype
    if args.dtype == "float16":
        print("Converting to float16 to save memory...")
        volume = volume.astype(np.float16)
    
    # Load SAM model
    print(f"\nLoading SAM model: {args.model_path}")
    sam_module = load_sam_module(args.model_path, device=args.device)
    
    # Segment along each axis
    seg_maps = {}
    axis_names = {0: "axial", 1: "sagittal", 2: "coronal"}
    
    for axis in args.axes:
        print(f"\n{'='*60}")
        print(f"Processing {axis_names[axis]} axis (axis={axis})")
        print(f"{'='*60}")
        
        seg_map = segment_axis_with_sam(
            volume,
            sam_module,
            axis,
            device=args.device,
            grid_stride=args.grid_stride,
            points_batch_size=args.points_batch_size,
            mode=args.mode,
            imgsz=args.imgsz,
            clear_cache_every=args.clear_cache_every,
        )
        seg_maps[axis] = seg_map
        
        # Clear memory after each axis
        if args.device == "cuda":
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
        # Save individual segmentations if requested
        if args.save_individual:
            outdir = Path(args.output).parent
            outname = Path(args.output).stem
            seg_path = outdir / f"{outname}_{axis_names[axis]}.tif"
            tifffile.imwrite(seg_path, (seg_map * 255).astype(np.uint8))
            print(f"Saved {axis_names[axis]} segmentation to {seg_path}")
    
    # Print statistics
    print_segmentation_stats(seg_maps, axis_names)
    
    # Fuse predictions
    print(f"\n{'='*60}")
    print(f"Fusing predictions using {args.fusion_mode} mode")
    print(f"{'='*60}")
    
    if args.fusion_mode == "majority":
        fused = fuse_majority(seg_maps)
    elif args.fusion_mode == "weighted":
        fused = fuse_weighted_average(seg_maps)
    elif args.fusion_mode == "uncertainty":
        fused = fuse_uncertainty_based(seg_maps)
    elif args.fusion_mode == "consensus":
        min_agreement = max(2, len(seg_maps) // 2 + 1)
        fused = fuse_consensus(seg_maps, min_agreement=min_agreement)
    
    n_fused = fused.sum()
    percentage = 100 * n_fused / fused.size
    print(f"Fused result: {n_fused:,} voxels ({percentage:.2f}%)")
    
    # Compute consistency score
    if args.save_consistency:
        print("\nComputing consistency map...")
        consistency = compute_consistency_score(seg_maps)
        outdir = Path(args.output).parent
        outname = Path(args.output).stem
        consistency_path = outdir / f"{outname}_consistency.tif"
        tifffile.imwrite(consistency_path, (consistency * 255).astype(np.uint8))
        print(f"Saved consistency map to {consistency_path}")
        
        # Print consistency statistics
        mean_consistency = consistency.mean()
        print(f"Mean consistency score: {mean_consistency:.4f}")
    
    # Post-process
    if not args.no_postprocess:
        print("\nPost-processing segmentation...")
        n_before = fused.sum()
        fused = postprocess_segmentation(
            fused,
            min_size=args.min_size,
            fill_holes=True,
        )
        n_after = fused.sum()
        print(f"  Before: {n_before:,} voxels")
        print(f"  After:  {n_after:,} voxels")
        print(f"  Change: {n_after - n_before:+,} voxels")
    
    # Save result
    print(f"\nSaving final segmentation to {args.output}")
    tifffile.imwrite(args.output, (fused * 255).astype(np.uint8))
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())