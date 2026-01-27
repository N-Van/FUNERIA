# src/train_sam_lora_25d.py
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from src.dataset_25d import Urn25DPairs
from src.lora import inject_lora_qkv, lora_state_dict
from src.losses import bce_dice_loss, interslice_coherence_loss
from src.sam_utils import forward_sam_best_logits
from src.lora import load_lora_state_dict


def read_pairs_list(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(";")
            pairs.append((a.strip(), b.strip()))
    return pairs


def set_trainable(sam, train_mask_decoder: bool):
    # freeze all
    for p in sam.parameters():
        p.requires_grad = False

    # unfreeze LoRA params
    for m in sam.modules():
        if m.__class__.__name__ == "LoRALinear":
            for p in m.parameters():
                p.requires_grad = True

    # optional: train mask decoder
    if train_mask_decoder:
        for p in sam.mask_decoder.parameters():
            p.requires_grad = True


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0
        self.best_state = None

    def step(self, metric, sam, epoch, args_dict):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.bad = 0
            self.best_state = {
                "lora": lora_state_dict(sam),
                "mask_decoder": {k: v.detach().cpu() for k, v in sam.mask_decoder.state_dict().items()} if args_dict["train_mask_decoder"] else None,
                "epoch": epoch,
                "val_dice": metric,
                "args": args_dict,
            }
            return False
        else:
            self.bad += 1
            return self.bad >= self.patience


@torch.no_grad()
def eval_val(sam, loader, resize, device, n_pos, n_neg):
    sam.eval()
    dices = []
    for x_z, y_z, x_zp1, y_zp1 in loader:
        # batch_size=1 conseillé en val (plus stable)
        x_z = x_z[0].to(device)       # [H,W,3]
        y_z = y_z[0].to(device)       # [H,W]
        x1  = x_zp1[0].to(device)
        y1  = y_zp1[0].to(device)

        logits_z  = forward_sam_best_logits(sam, resize, x_z, y_z, device, n_pos=n_pos, n_neg=n_neg)
        logits_z1 = forward_sam_best_logits(sam, resize, x1,  y1,  device, n_pos=n_pos, n_neg=n_neg)

        # dice mean (z et z+1)
        dz  = (2*(torch.sigmoid(logits_z )*y_z ).sum() + 1e-6) / (torch.sigmoid(logits_z ).sum() + y_z.sum() + 1e-6)
        dz1 = (2*(torch.sigmoid(logits_z1)*y1 ).sum() + 1e-6) / (torch.sigmoid(logits_z1).sum() + y1.sum() + 1e-6)
        dices.append(((dz + dz1) * 0.5).item())

    return float(np.mean(dices)) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_list", type=str, required=True)
    ap.add_argument("--val_list", type=str, required=True)

    ap.add_argument("--sam_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--sam_ckpt", type=str, required=True)

    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    ap.add_argument("--crop_size", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--points_pos", type=int, default=8)   # augmente pour simuler segment-all
    ap.add_argument("--points_neg", type=int, default=8)

    ap.add_argument("--lambda_coh", type=float, default=0.2)
    ap.add_argument("--coh_mode", type=str, default="gt_union", choices=["gt_union", "all"])
    ap.add_argument("--train_mask_decoder", action="store_true")

    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out_dir", type=str, default="ckpt_sam_lora_25d")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- SAM
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(device)
    sam.train()

    # ---- LoRA
    replaced = inject_lora_qkv(sam.image_encoder, r=args.r, alpha=args.alpha, dropout=args.lora_dropout)
    print(f"[LoRA] replaced attn.qkv: {replaced}")

    set_trainable(sam, train_mask_decoder=args.train_mask_decoder)

    # ---- Data (concat multi-volumes)
    train_pairs = read_pairs_list(args.train_list)
    val_pairs   = read_pairs_list(args.val_list)

    ds_train = []
    for k, (vp, mp) in enumerate(train_pairs):
        ds_train.append(Urn25DPairs(vp, mp, crop_size=args.crop_size, seed=1000 + k, random_crop=True))
    ds_train = ConcatDataset(ds_train)

    ds_val = []
    for k, (vp, mp) in enumerate(val_pairs):
        # val sans crop random => random_crop=False et crop_size ignoré
        ds_val.append(Urn25DPairs(vp, mp, crop_size=args.crop_size, seed=2000 + k, random_crop=False))
    ds_val = ConcatDataset(ds_val)

    dl_tr = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=args.num_workers)

    resize = ResizeLongestSide(sam.image_encoder.img_size)

    # ---- Optim
    params = [p for p in sam.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    stopper = EarlyStopping(patience=args.patience, min_delta=1e-4)

    for epoch in range(1, args.epochs + 1):
        sam.train()
        losses = []

        for x_z, y_z, x_zp1, y_zp1 in dl_tr:
            # shapes: [B,H,W,3] et [B,H,W]
            x_z   = x_z.to(device)
            y_z   = y_z.to(device)
            x_zp1 = x_zp1.to(device)
            y_zp1 = y_zp1.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                batch_loss = 0.0
                B = x_z.shape[0]

                for b in range(B):
                    logits0 = forward_sam_best_logits(
                        sam, resize,
                        x_z[b], y_z[b],
                        device,
                        n_pos=args.points_pos,
                        n_neg=args.points_neg,
                        multimask_output=True,
                    )
                    logits1 = forward_sam_best_logits(
                        sam, resize,
                        x_zp1[b], y_zp1[b],
                        device,
                        n_pos=args.points_pos,
                        n_neg=args.points_neg,
                        multimask_output=True,
                    )

                    seg0 = bce_dice_loss(logits0, y_z[b])
                    seg1 = bce_dice_loss(logits1, y_zp1[b])
                    seg = 0.5 * (seg0 + seg1)

                    coh = interslice_coherence_loss(
                        logits0, logits1, y_z[b], y_zp1[b],
                        mode=args.coh_mode
                    )

                    batch_loss = batch_loss + (seg + args.lambda_coh * coh)

                batch_loss = batch_loss / B

            scaler.scale(batch_loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(batch_loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else 0.0

        # ---- VAL (early stopping sur dice)
        val_dice = eval_val(sam, dl_va, resize, device, args.points_pos, args.points_neg)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_dice={val_dice:.4f}")

        # save last
        last_path = os.path.join(args.out_dir, "last.pt")
        torch.save({
            "lora": lora_state_dict(sam),
            "mask_decoder": sam.mask_decoder.state_dict() if args.train_mask_decoder else None,
            "epoch": epoch,
            "val_dice": val_dice,
            "args": vars(args),
        }, last_path)

        stop = stopper.step(val_dice, sam, epoch, vars(args))
        if stop:
            print(f"[EarlyStopping] stop at epoch {epoch}, best={stopper.best:.4f}")
            break

    best_path = os.path.join(args.out_dir, "best.pt")
    torch.save(stopper.best_state, best_path)
    print(f"[Saved] best -> {best_path} (val_dice={stopper.best:.4f})")


if __name__ == "__main__":
    main()
