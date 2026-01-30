# src/sam_utils.py
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

# Pixel mean/std de SAM (Meta)
SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
SAM_PIXEL_STD  = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

def mask_to_box(mask: torch.Tensor, jitter: int = 10):
    """
    mask: (H,W) float {0,1}
    retourne box XYXY en pixels (float) + jitter.
    """
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0:
        # aucun pixel -> box full image (fallback)
        H, W = mask.shape
        return torch.tensor([0, 0, W-1, H-1], device=mask.device).float()

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # jitter léger (augmentation)
    if jitter > 0:
        H, W = mask.shape
        x0 = (x0 - torch.randint(0, jitter+1, (1,), device=mask.device)).clamp(0, W-1)
        y0 = (y0 - torch.randint(0, jitter+1, (1,), device=mask.device)).clamp(0, H-1)
        x1 = (x1 + torch.randint(0, jitter+1, (1,), device=mask.device)).clamp(0, W-1)
        y1 = (y1 + torch.randint(0, jitter+1, (1,), device=mask.device)).clamp(0, H-1)

    return torch.stack([x0, y0, x1, y1]).float()

def preprocess_for_sam(x_hwc: torch.Tensor, sam_model, device):
    """
    x_hwc: (H,W,3) float [0,1]
    retourne image_tensor: (1,3,1024,1024) float (normalisé/paddé),
            transform: ResizeLongestSide, resized_hw (h,w)
    """
    img_size = sam_model.image_encoder.img_size  # 1024
    transform = ResizeLongestSide(img_size)

    # SAM attend "image" en uint8 0..255 typiquement
    x_np = (x_hwc.clamp(0,1).cpu().numpy() * 255.0).astype(np.uint8)
    x_rs = transform.apply_image(x_np)  # H'W'3

    x_t = torch.as_tensor(x_rs, device=device).permute(2,0,1).contiguous().float()

    # normalize
    mean = SAM_PIXEL_MEAN.to(device)
    std  = SAM_PIXEL_STD.to(device)
    x_t = (x_t - mean) / std

    # pad to square img_size
    h, w = x_t.shape[-2:]
    pad_h = img_size - h
    pad_w = img_size - w
    x_t = F.pad(x_t, (0, pad_w, 0, pad_h), value=0.0)  # (3,1024,1024)

    return x_t.unsqueeze(0), transform, (h, w)

def preprocess_mask_for_sam(mask_hw: torch.Tensor, transform, img_size: int, device):
    """
    mask_hw: (H,W) float {0,1}
    retourne mask_256: (1,1,256,256)
    """
    mask_np = (mask_hw.cpu().numpy() > 0.5).astype(np.uint8) * 255
    # resize comme l'image (sans padding dans apply_image, mais on reproduit padding ensuite)
    m_rs = transform.apply_image(mask_np)  # H'W' (mais retourne parfois H'W'3 -> on gère)
    if m_rs.ndim == 3:
        m_rs = m_rs[...,0]
    m_t = torch.as_tensor((m_rs > 0).astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)  # 1,1,h,w

    # pad à 1024 comme l'image
    h, w = m_t.shape[-2:]
    pad_h = img_size - h
    pad_w = img_size - w
    m_t = F.pad(m_t, (0, pad_w, 0, pad_h), value=0.0)  # 1,1,1024,1024

    # downsample à 256 (résolution low-res du decoder)
    m_256 = F.interpolate(m_t, size=(256,256), mode="nearest")
    return m_256

def forward_sam_with_box(sam_model, x_hwc, y_hw, device, jitter=10):
    """
    Retourne logits_lowres: (1,1,256,256) (non sigmoid)
    """
    img_size = sam_model.image_encoder.img_size  # 1024
    x_in, transform, _ = preprocess_for_sam(x_hwc, sam_model, device)
    y_256 = preprocess_mask_for_sam(y_hw, transform, img_size, device)

    # box prompt depuis GT (sur coords originales), puis transform vers resized coords
    box = mask_to_box(y_hw.to(device), jitter=jitter).unsqueeze(0)  # (1,4) XYXY
    box_np = box.detach().cpu().numpy()
    box_rs = transform.apply_boxes(box_np, (y_hw.shape[0], y_hw.shape[1]))  # coords sur resized (avant padding)
    box_rs = torch.as_tensor(box_rs, device=device).float()

    # encode image
    image_embeddings = sam_model.image_encoder(x_in)

    # prompt embeddings
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_rs,
        masks=None,
    )

    low_res_masks, iou_pred = sam_model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    # low_res_masks: (1,1,256,256) logits
    return low_res_masks, y_256
