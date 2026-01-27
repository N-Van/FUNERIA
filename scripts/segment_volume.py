from pathlib import Path

import numpy as np
import tifffile as tiff
import torch

from src.data.one_urn_datamodule import OneUrnDataModule
from src.models.sam_module import SAM3DModuleLinear


def compute_urna_mask(gray_hw: np.ndarray) -> np.ndarray:
    # True si intensité > 60 (sur intensités brutes)
    return gray_hw.astype(np.float32) > 60.0


def masks_to_binary(masks_bool: np.ndarray) -> np.ndarray | None:
    """(N,H,W) bool -> (H,W) bool (union)"""
    if masks_bool is None or masks_bool.size == 0 or masks_bool.shape[0] == 0:
        return None
    return np.any(masks_bool, axis=0)


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("outputs/sam_binary")
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: on lit le TIFF brut pour le seuil > 60
    raw_vol = tiff.imread("data/urne_640.tif").astype(np.float32)  # (Z,H,W)

    dm = OneUrnDataModule(
        filename="data/urne_640.tif",
        ground_truth_filename="data/urne_truth_640.tif",
        slice_jump=1,
        slice_image_size=640,
        train_val_test_split=(0, 0, 1),
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("test")

    projections, _ = next(iter(dm.test_dataloader()))  # (Z,3,H,W) NORMALISE 0..1
    Z, _, H, W = projections.shape
    print("projections:", projections.shape)

    # check shape match
    if raw_vol.shape != (Z, H, W):
        raise ValueError(f"raw_vol shape {raw_vol.shape} != (Z,H,W) {(Z,H,W)}")

    model = (
        SAM3DModuleLinear(
            sam_checkpoint="weights/sam_b.pt",
            points_stride=32,
            points_batch_size=25,
        )
        .eval()
        .to(device)
    )

    stride = int(model.hparams["points_stride"])
    bsz = int(model.hparams["points_batch_size"])

    seg_3d = np.zeros((Z, H, W), dtype=np.uint8)  # 0/255

    for z in range(Z):
        frame = projections[z : z + 1].to(device)  # (1,3,H,W) pour SAM

        # MASK > 60 SUR LE TIFF BRUT
        gray_raw = raw_vol[z]  # (H,W) en 0..255
        urna_mask_np = compute_urna_mask(gray_raw)

        masks_f, info = model.infer_one_slice(
            frame,
            urna_mask=urna_mask_np,
            mode="auto",
            # grid_stride=stride,
            points_batch_size=bsz,
        )  # (N,H,W) bool

        bin2d = masks_to_binary(masks_f)
        if bin2d is not None:
            seg_3d[z] = bin2d.astype(np.uint8) * 255

        if (z % 25) == 0:
            print(
                f"z={z}/{Z}  raw={info['n_raw']} keep={info['n_keep']} points={info['n_points']}"
            )

    tiff.imwrite(out_dir / "seg_binary_3d.tif", seg_3d)
    print("Saved:", out_dir / "seg_binary_3d.tif")


if __name__ == "__main__":
    main()
