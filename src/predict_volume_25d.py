# src/predict_volume_25d_improved.py
import os
import argparse
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from lora import inject_lora_qkv, load_lora_state_dict


# -----------------------
# SAM normalization constants
# -----------------------
SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
SAM_PIXEL_STD  = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def sample_percentiles(vol_memmap, n_slices=32, seed=0):
    rng = np.random.RandomState(seed)
    D = vol_memmap.shape[0]
    idx = rng.choice(np.arange(D), size=min(n_slices, D), replace=False)
    samp = np.concatenate([vol_memmap[i].ravel() for i in idx], axis=0).astype(np.float32)
    p1 = float(np.percentile(samp, 1))
    p99 = float(np.percentile(samp, 99))
    return p1, p99


def norm_slice(s2d, p1, p99):
    x = s2d.astype(np.float32)
    x = (x - p1) / max(1e-6, (p99 - p1))
    return np.clip(x, 0.0, 1.0)


def make_25d(vol_ax0, z, p1, p99):
    # vol_ax0: (D,H,W) where D is the sweeping axis
    D = vol_ax0.shape[0]
    z0 = max(0, z - 1)
    z1 = z
    z2 = min(D - 1, z + 1)
    s0 = norm_slice(vol_ax0[z0], p1, p99)
    s1 = norm_slice(vol_ax0[z1], p1, p99)
    s2 = norm_slice(vol_ax0[z2], p1, p99)
    return np.stack([s0, s1, s2], axis=-1)  # HWC float [0,1]


def bbox_full(H, W):
    return np.array([0, 0, W - 1, H - 1], dtype=np.float32)


def expand_box(box, H, W, pad=20):
    x0, y0, x1, y1 = box.tolist()
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def box_from_prob(prob_hw, thr=0.20, pad=15):
    """
    prob_hw: float [0,1] (H,W)
    Use a low threshold to get a stable box even when final mask is weak.
    """
    m = (prob_hw >= thr)
    ys, xs = np.where(m)
    H, W = prob_hw.shape
    if xs.size == 0:
        return bbox_full(H, W)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad); y1 = min(H - 1, y1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def preprocess_for_sam(x_hwc01: np.ndarray, device: str):
    """
    x_hwc01: HWC float in [0,1], 3 channels
    Returns:
      x_in: (1,3,1024,1024) normalized for SAM
      transform: ResizeLongestSide(1024)
      (h_rs,w_rs): resized before padding
      (H,W): original
    """
    assert x_hwc01.ndim == 3 and x_hwc01.shape[2] == 3
    H, W = x_hwc01.shape[0], x_hwc01.shape[1]

    transform = ResizeLongestSide(1024)

    x_u8 = (np.clip(x_hwc01, 0, 1) * 255.0).astype(np.uint8)
    x_rs = transform.apply_image(x_u8)  # H'W'3
    h_rs, w_rs = x_rs.shape[0], x_rs.shape[1]

    x_t = torch.from_numpy(x_rs).to(device).float().permute(2, 0, 1).unsqueeze(0)  # 1,3,h',w'
    pad_h = 1024 - h_rs
    pad_w = 1024 - w_rs
    x_t = F.pad(x_t, (0, pad_w, 0, pad_h), value=0)

    mean = SAM_PIXEL_MEAN.to(device)
    std = SAM_PIXEL_STD.to(device)
    x_t = (x_t - mean) / std
    return x_t, transform, (h_rs, w_rs), (H, W)


@torch.no_grad()
def predict_prob_with_box(sam, x25_hwc01, box_xyxy, device, amp=False):
    """
    Returns prob map at original size: (H,W) float32
    """
    H, W = x25_hwc01.shape[0], x25_hwc01.shape[1]
    x_in, transform, (h_rs, w_rs), _ = preprocess_for_sam(x25_hwc01, device)

    # box -> resized coords (before padding)
    box_rs = transform.apply_boxes(box_xyxy[None, :], (H, W))
    box_rs = torch.as_tensor(box_rs, device=device).float()

    with torch.amp.autocast("cuda", enabled=amp and (device == "cuda")):
        emb = sam.image_encoder(x_in)
        sparse, dense = sam.prompt_encoder(points=None, boxes=box_rs, masks=None)
        low_res_logits, _ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )  # (1,1,256,256)

        logits_1024 = F.interpolate(low_res_logits, size=(1024, 1024), mode="bilinear", align_corners=False)
        prob_1024 = torch.sigmoid(logits_1024)

        # remove padding -> resized -> original
        prob_rs = prob_1024[..., :h_rs, :w_rs]
        prob_orig = F.interpolate(prob_rs, size=(H, W), mode="bilinear", align_corners=False)

    return prob_orig.squeeze().float().cpu().numpy()


def binary_closing(mask_hw: np.ndarray, k=5, it=1):
    """
    Simple morphological closing: dilate then erode (CPU, torch).
    mask_hw: uint8 0/1
    """
    x = torch.from_numpy(mask_hw[None, None].astype(np.float32))
    pad = k // 2
    for _ in range(it):
        # dilate
        x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        # erode via minpool
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)
    return (x.squeeze().numpy() > 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vol", default="resized.tiff")
    ap.add_argument("--ckpt", default="sam_vit_b_01ec64.pth")
    ap.add_argument("--lora_ckpt", required=True)
    ap.add_argument("--out_tif", default="runs/urne_sam_lora_25d/pred_improved.tiff")

    ap.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--axis", type=int, default=0, choices=[0, 1, 2], help="0=z, 1=y, 2=x (sweep axis)")
    ap.add_argument("--stride", type=int, default=1, help="process every stride slice (others copied)")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--thr", type=float, default=0.50, help="final mask threshold")
    ap.add_argument("--thr_box", type=float, default=0.20, help="low thr for box estimation from prob")
    ap.add_argument("--box_pad", type=int, default=15, help="pad for box from prob")
    ap.add_argument("--expand", type=int, default=25, help="expand previous box on recovery")
    ap.add_argument("--min_area", type=int, default=200, help="if mask area < min_area => recovery")
    ap.add_argument("--ema", type=float, default=0.80, help="EMA factor on prob (0 disables if <0)")
    ap.add_argument("--closing_k", type=int, default=0, help="0=off, else kernel size for closing (e.g., 5)")
    ap.add_argument("--closing_it", type=int, default=1)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z_max", type=int, default=-1, help="debug: limit slices, -1=all")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> predict_volume_25d_improved.py started")
    print(f"[Device] {device}")

    vol_path = os.path.join(args.data_dir, args.vol)
    ckpt_path = os.path.join(args.data_dir, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt

    vol = tiff.memmap(vol_path)
    if vol.ndim != 3:
        raise ValueError(f"Volume must be 3D (D,H,W) got {vol.shape}")

    # Move sweep axis to axis 0 so we reuse same loop code
    vol_ax = np.moveaxis(vol, args.axis, 0)  # (D_sweep, H2, W2)
    D, H, W = vol_ax.shape
    z_stop = D if args.z_max < 0 else min(D, args.z_max)
    print(f"[Volume] orig={vol.shape} sweep_axis={args.axis} working_shape={vol_ax.shape} z_stop={z_stop}")

    p1, p99 = sample_percentiles(vol_ax, n_slices=32, seed=args.seed)
    print(f"[Norm] p1={p1:.3f} p99={p99:.3f}")

    # Load SAM
    sam = sam_model_registry[args.model_type](checkpoint=ckpt_path).to(device)
    sam.eval()

    # Inject LoRA
    replaced = inject_lora_qkv(sam.image_encoder, r=args.r, alpha=args.alpha, dropout=args.dropout)
    print(f"[LoRA] injected qkv layers: {replaced}")

    pack = torch.load(args.lora_ckpt, map_location="cpu")
    lora_sd = pack["lora"] if isinstance(pack, dict) and "lora" in pack else pack
    load_lora_state_dict(sam.image_encoder, lora_sd, strict=True)
    sam.to(device)
    sam.eval()
    print("[LoRA] weights loaded OK")

    out_ax = np.zeros((z_stop, H, W), dtype=np.uint8)

    prev_box = bbox_full(H, W)
    prob_ema = None

    empty = 0
    recovered = 0

    z_list = list(range(0, z_stop, max(1, args.stride)))
    for zi, z in enumerate(tqdm(z_list, desc="Predict 3D (improved)")):
        x25 = make_25d(vol_ax, z, p1, p99)

        # ---- pass 1: use prev_box
        prob = predict_prob_with_box(sam, x25, prev_box, device, amp=args.amp)

        # EMA smoothing across slices
        if args.ema >= 0.0:
            if prob_ema is None:
                prob_ema = prob.copy()
            else:
                prob_ema = args.ema * prob_ema + (1.0 - args.ema) * prob
            prob_use = prob_ema
        else:
            prob_use = prob

        mask = (prob_use >= args.thr).astype(np.uint8)

        # ---- recovery criteria
        if int(mask.sum()) < args.min_area:
            # try expanded box
            box2 = expand_box(prev_box, H, W, pad=args.expand)
            prob2 = predict_prob_with_box(sam, x25, box2, device, amp=args.amp)
            mask2 = (prob2 >= args.thr).astype(np.uint8)

            if int(mask2.sum()) >= args.min_area:
                prob = prob2
                mask = mask2
                prev_box = box_from_prob(prob, thr=args.thr_box, pad=args.box_pad)
                recovered += 1
            else:
                # try full box + lower "box thr" update
                prob3 = predict_prob_with_box(sam, x25, bbox_full(H, W), device, amp=args.amp)
                mask3 = (prob3 >= args.thr).astype(np.uint8)

                if int(mask3.sum()) >= args.min_area:
                    prob = prob3
                    mask = mask3
                    prev_box = box_from_prob(prob, thr=args.thr_box, pad=args.box_pad)
                    recovered += 1
                else:
                    empty += 1
                    # keep previous box (don’t collapse to full every time)
                    # but still store empty mask
                    prev_box = expand_box(prev_box, H, W, pad=args.expand)

        else:
            # normal update: compute box from LOW thr prob (more stable than mask)
            prev_box = box_from_prob(prob_use, thr=args.thr_box, pad=args.box_pad)

        # optional closing to reduce holes
        if args.closing_k and args.closing_k > 1:
            mask = binary_closing(mask, k=args.closing_k, it=args.closing_it)

        # write with stride fill
        if args.stride <= 1:
            out_ax[z] = mask
        else:
            z_end = min(z_stop, z + args.stride)
            out_ax[z:z_end] = mask  # simple copy to intermediate slices

        if (z % 50) == 0:
            print(f"[z={z}] area={int(mask.sum())} empty={empty} recovered={recovered} box={prev_box.astype(int).tolist()}")

    # Move axis back to original orientation
    out_full = np.moveaxis(out_ax, 0, args.axis)  # back to (D,H,W) like input

    os.makedirs(os.path.dirname(args.out_tif) or ".", exist_ok=True)
    tiff.imwrite(args.out_tif, (out_full * 255).astype(np.uint8), compression="zlib")
    print(f"[Saved] {args.out_tif}")
    print(f"[Done] empty={empty} recovered={recovered} total_slices={z_stop} stride={args.stride}")


if __name__ == "__main__":
    main()
