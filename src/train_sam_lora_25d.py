# src/train_urne_sam_lora_25d.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from dataset_25d import Urn25DPairs
from lora import inject_lora_qkv, lora_state_dict
from sam_utils import forward_sam_with_box

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1,2,3))
    den = (probs + targets).sum(dim=(1,2,3)) + eps
    return (1.0 - num / den).mean()

def bce_loss_from_logits(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

def soft_iou(probs1, probs2, eps=1e-6):
    # probs: (B,1,H,W)
    inter = (probs1 * probs2).sum(dim=(1,2,3))
    union = (probs1 + probs2 - probs1 * probs2).sum(dim=(1,2,3)) + eps
    return (inter / union)

def hard_iou(mask1, mask2, eps=1e-6):
    # mask: (B,1,H,W) float 0/1
    inter = (mask1 * mask2).sum(dim=(1,2,3))
    union = (mask1 + mask2 - mask1 * mask2).sum(dim=(1,2,3)) + eps
    return inter / union

def load_sam(checkpoint_path: str, model_type="vit_b", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=None).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # essaie de récupérer un state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format not understood. Provide SAM-compatible state_dict.")

    missing, unexpected = sam.load_state_dict(ckpt, strict=False)
    print(f"[SAM] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    return sam

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def only_lora_trainable(model):
    trainable = []
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False
    return trainable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", type=str)
    ap.add_argument("--vol", default="resized.tiff", type=str)
    ap.add_argument("--mask", default="urne_truth_640.tif", type=str)
    ap.add_argument("--ckpt", default="sam_b.pt", type=str)
    ap.add_argument("--model_type", default="vit_b", type=str)
    ap.add_argument("--out", default="runs/urne_sam_lora_25d", type=str)

    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--batch", default=2, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--wd", default=0.01, type=float)
    ap.add_argument("--num_workers", default=4, type=int)

    ap.add_argument("--crop", default=640, type=int)     # 640 = full, sinon 512 etc.
    ap.add_argument("--jitter", default=10, type=int)    # jitter box prompt
    ap.add_argument("--lam_cons", default=0.1, type=float)  # poids cohérence
    ap.add_argument("--r", default=8, type=int)
    ap.add_argument("--alpha", default=16, type=int)
    ap.add_argument("--dropout", default=0.0, type=float)

    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    vol_path  = os.path.join(args.data_dir, args.vol)
    mask_path = os.path.join(args.data_dir, args.mask)
    ckpt_path = os.path.join(args.data_dir, args.ckpt)

    # data
    ds = Urn25DPairs(vol_path, mask_path, crop_size=args.crop, seed=0)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # model
    sam = load_sam(ckpt_path, model_type=args.model_type, device=device)
    freeze_all(sam)

    replaced = inject_lora_qkv(sam.image_encoder, r=args.r, alpha=args.alpha, dropout=args.dropout)
    print(f"[LoRA] replaced qkv layers: {replaced}")

    trainable = only_lora_trainable(sam)
    print(f"[Trainable params] {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    sam.train()

    for ep in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {ep}/{args.epochs}")
        running = {"sup": 0.0, "cons": 0.0, "total": 0.0}

        for (x_z, y_z, x_zp1, y_zp1) in pbar:
            # x_*: (B,H,W,3) float [0,1], y_*: (B,H,W)
            x_z   = x_z.to(device, non_blocking=True)
            y_z   = y_z.to(device, non_blocking=True)
            x_zp1 = x_zp1.to(device, non_blocking=True)
            y_zp1 = y_zp1.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # forward slice z
                # On boucle sur le batch car forward_sam_with_box est écrit pour 1 sample (simple + sûr).
                logits_list = []
                gt256_list = []
                logits_list_p1 = []
                gt256_list_p1 = []

                for b in range(x_z.shape[0]):
                    logits_z, gt_z_256 = forward_sam_with_box(sam, x_z[b], y_z[b], device, jitter=args.jitter)
                    logits_p1, gt_p1_256 = forward_sam_with_box(sam, x_zp1[b], y_zp1[b], device, jitter=args.jitter)
                    logits_list.append(logits_z)
                    gt256_list.append(gt_z_256)
                    logits_list_p1.append(logits_p1)
                    gt256_list_p1.append(gt_p1_256)

                logits_z   = torch.cat(logits_list, dim=0)       # (B,1,256,256)
                gt_z_256   = torch.cat(gt256_list, dim=0)
                logits_zp1 = torch.cat(logits_list_p1, dim=0)
                gt_zp1_256 = torch.cat(gt256_list_p1, dim=0)

                # supervised loss (sur z et z+1)
                loss_sup = (
                    bce_loss_from_logits(logits_z, gt_z_256) + dice_loss_from_logits(logits_z, gt_z_256)
                    + bce_loss_from_logits(logits_zp1, gt_zp1_256) + dice_loss_from_logits(logits_zp1, gt_zp1_256)
                ) * 0.5

                # inter-slice consistency: Soft-IoU entre probas
                p_z   = torch.sigmoid(logits_z)
                p_zp1 = torch.sigmoid(logits_zp1)
                sim_pred = soft_iou(p_z, p_zp1)  # (B,)

                # pondération par IoU GT (si GT change => on force moins)
                iou_gt = hard_iou(gt_z_256, gt_zp1_256).detach()  # (B,)
                loss_cons = (1.0 - sim_pred) * iou_gt
                loss_cons = loss_cons.mean()

                loss = loss_sup + args.lam_cons * loss_cons

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running["sup"] += float(loss_sup.item())
            running["cons"] += float(loss_cons.item())
            running["total"] += float(loss.item())
            n = 1
            pbar.set_postfix(
                sup=running["sup"]/n,
                cons=running["cons"]/n,
                total=running["total"]/n
            )

        # save LoRA weights each epoch
        ckpt_out = os.path.join(args.out, f"lora_epoch_{ep:03d}.pt")
        torch.save({
            "epoch": ep,
            "lora": lora_state_dict(sam.image_encoder),
            "args": vars(args),
        }, ckpt_out)
        print(f"[Saved] {ckpt_out}")

if __name__ == "__main__":
    main()
