## Fine-tuning SAM for 3D Urn Segmentation (LoRA – 2.5D)

### Objective
We fine-tune **Segment Anything (SAM)** to perform **automatic 3D segmentation** of funerary urns from micro-CT volumes  
(**TIFF 640×640×640**), **without user interaction** (no points, no clicks), while enforcing **inter-slice coherence**.

SAM is originally a 2D, interaction-driven model.  
Our goal is to adapt it to **volumetric data** and **fully automatic inference**.

---

## Method Overview

### 1. Input Representation (2.5D)
Each slice `z` is represented using a **2.5D context**:
- Channel R → slice `z-1`
- Channel G → slice `z`
- Channel B → slice `z+1`

This allows SAM to capture **local 3D continuity** while remaining compatible with its 2D architecture. We could use differents Delta (here we used Delta = 1).

---

### 2. Fine-tuning Strategy (LoRA)

We use **Low-Rank Adaptation (LoRA)** to fine-tune SAM efficiently.

- Backbone: **SAM ViT-B**
- LoRA applied **only to the QKV projection layers** of the image encoder
- Original SAM weights are **frozen**
- Only a small number of parameters are trained

**LoRA configuration:**
- Rank `r = 8`
- Scaling factor `α = 16`
- Dropout `= 0.0`
- Trainable parameters ≈ **295k**

This preserves the generalization ability of SAM while adapting it to the urn domain.

---

### 3. Supervision and Losses

For two consecutive slices `(z, z+1)`:

**Supervised loss (per slice):**
- Binary Cross-Entropy


**Inter-slice consistency loss:**
- L1 distance between soft predictions of slice `z` and `z+1`
- Edge-weighted to focus on object boundaries

**Total loss:**
L = L_supervised + λ · L_consistency

the supervised loss using a BCE betwen the logits, and the consistency loss is calculated with a (1-Dice) between the ground truth and our prediction so that 
where `λ` is progressively increased during training (warm-up).

---

### 4. Training Setup

- Training performed on **random crops (512×512)** to increase data diversity
- Same crop location applied to `(z, z+1)` to preserve spatial consistency
- One urn volume used for training
- Optimization: AdamW
- Mixed precision (AMP) supported

---

## Training Command

```bash
python src/train_sam_lora_25d.py \
  --data_dir data \
  --vol resized.tiff \
  --mask urne_truth_640.tif \
  --ckpt sam_vit_b_01ec64.pth \
  --model_type vit_b \
  --epochs 20 \
  --batch 2 \
  --crop 512 \
  --r 8 \
  --alpha 16 \
  --lam_cons 0.1 \
  --amp \
  --out runs/urne_sam_lora_25d
```
Outputs:

LoRA checkpoints: runs/urne_sam_lora_25d/lora_epoch_XXX.pt

Final model used: lora_epoch_020.pt

Inference on a New Urn Volume
Inference is fully automatic:No points, No clicks

Bounding box is propagated slice-to-slice to stabilize predictions

## Inference Command
```bash
python src/predict_volume_25d.py \
  --data_dir data \
  --vol new_urne_resized.tiff \
  --ckpt sam_vit_b_01ec64.pth \
  --lora_ckpt runs/urne_sam_lora_25d/lora_epoch_020.pt \
  --model_type vit_b \
  --thr 0.5 \
  --out_tif runs/new_urne_pred_lora.tiff
```
Output:

3D binary segmentation (uint8, values {0,255})

Shape: (D, H, W)

## Evaluation
Volumetric Metrics :Dice, IoU, Precision / Recall
```bash
python src/compare_tiff_metrics.py \
  --pred runs/new_urne_pred_lora.tiff \
  --gt data/new_urne_groundtruth.tif
```

## Key Takeaway
Fine-tuning SAM with LoRA and a 2.5D formulation with a consistency_loss should enable coherent 3D segmentation from a model originally designed for interactive 2D tasks, while keeping training efficient and fully automatic.Try modifing this key parametres to improve our finetuned model : r (the rank of matrix ), alpha (Scaling factor), delta(steps), the number of epochs (add early stopping on the validation set), augment the number of datas to train the model, try finetuning the mask decoder in stead of the backbone, ...  

