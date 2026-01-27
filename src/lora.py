# src/lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Linear gelé + delta LoRA: W_eff = W0 + (B @ A) * scaling
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # poids de base gelés (copie)
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(dropout)

        # IMPORTANT: LoRA params sur le MEME device/dtype que weight
        device = self.weight.device
        dtype = self.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=dtype))

        # init : A kaiming, B zéro => delta ~ 0 au début
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A, None)
        lora = F.linear(lora, self.lora_B, None) * self.scaling
        return base + lora


def _set_module(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora_qkv(module_root: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """
    Remplace toutes les Linear '...attn.qkv' par LoRALinear.
    À appeler sur sam.image_encoder (ou sam.image_encoder directement).
    """
    replaced = 0
    for name, module in list(module_root.named_modules()):
        if name.endswith("attn.qkv") and isinstance(module, nn.Linear):
            _set_module(module_root, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def lora_state_dict(model: nn.Module) -> dict:
    """Sauve uniquement les poids LoRA (A/B)."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict, strict: bool = True):
    """Recharge les poids LoRA (A/B) dans un modèle déjà patché."""
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            kA, kB = f"{name}.lora_A", f"{name}.lora_B"
            if kA in sd and kB in sd:
                module.lora_A.data.copy_(sd[kA].to(module.lora_A.device).to(module.lora_A.dtype))
                module.lora_B.data.copy_(sd[kB].to(module.lora_B.device).to(module.lora_B.dtype))
            else:
                missing.append(name)
    if strict and missing:
        raise RuntimeError(f"LoRA keys missing for modules: {missing}")
