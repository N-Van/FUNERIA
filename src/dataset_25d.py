# src/dataset_25d.py
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

def _to_binary(mask2d: np.ndarray) -> np.ndarray:
    return (mask2d > 0).astype(np.uint8)

def _normalize_uint_like(vol2d: np.ndarray, p1: float, p99: float) -> np.ndarray:
    x = vol2d.astype(np.float32)
    x = (x - p1) / max(1e-6, (p99 - p1))
    x = np.clip(x, 0.0, 1.0)
    return x

def _sample_percentiles(volume_memmap, n_slices=32, seed=0):
    rng = np.random.RandomState(seed)
    D = volume_memmap.shape[0]
    idx = rng.choice(np.arange(D), size=min(n_slices, D), replace=False)
    samp = np.concatenate([volume_memmap[i].ravel() for i in idx], axis=0).astype(np.float32)
    p1 = float(np.percentile(samp, 1))
    p99 = float(np.percentile(samp, 99))
    return p1, p99

class Urn25DPairs(Dataset):
    """
    Retourne un couple (z, z+1) :
      x_z   = [z-1, z, z+1]
      x_zp1 = [z, z+1, z+2]
      y_z   = mask(z)
      y_zp1 = mask(z+1)
    """
    def __init__(
        self,
        vol_path: str,
        mask_path: str,
        crop_size: int = 640,
        seed: int = 0,
        random_crop: bool = True,
    ):
        self.vol = tiff.memmap(vol_path)    # (D,H,W)
        self.msk = tiff.memmap(mask_path)   # (D,H,W) ou (H,W,D)

        if self.msk.ndim != 3:
            raise ValueError(f"Mask tif must be 3D, got shape={self.msk.shape}")

        if self.msk.shape[0] != self.vol.shape[0] and self.msk.shape[-1] == self.vol.shape[0]:
            self.msk = np.moveaxis(self.msk, -1, 0)

        assert self.vol.shape[0] == self.msk.shape[0], f"Depth mismatch: vol {self.vol.shape}, msk {self.msk.shape}"
        self.D, self.H, self.W = self.vol.shape

        self.crop = int(crop_size)
        self.rng = np.random.RandomState(seed)
        self.random_crop = bool(random_crop)

        self.p1, self.p99 = _sample_percentiles(self.vol, n_slices=32, seed=seed)
        self.indices = list(range(self.D - 1))

    def __len__(self):
        return len(self.indices)

    def _get_slice(self, z: int) -> np.ndarray:
        z = int(np.clip(z, 0, self.D - 1))
        return self.vol[z]

    def _get_mask(self, z: int) -> np.ndarray:
        z = int(np.clip(z, 0, self.D - 1))
        return _to_binary(self.msk[z])

    def _make_25d(self, z: int) -> np.ndarray:
        s0 = _normalize_uint_like(self._get_slice(z - 1), self.p1, self.p99)
        s1 = _normalize_uint_like(self._get_slice(z),     self.p1, self.p99)
        s2 = _normalize_uint_like(self._get_slice(z + 1), self.p1, self.p99)
        return np.stack([s0, s1, s2], axis=-1)  # HWC

    def _random_crop(self, x, y, x2, y2):
        if (not self.random_crop) or self.crop >= self.H or self.crop >= self.W:
            return x, y, x2, y2
        top = self.rng.randint(0, self.H - self.crop + 1)
        left = self.rng.randint(0, self.W - self.crop + 1)
        x  = x [top:top+self.crop, left:left+self.crop, :]
        x2 = x2[top:top+self.crop, left:left+self.crop, :]
        y  = y [top:top+self.crop, left:left+self.crop]
        y2 = y2[top:top+self.crop, left:left+self.crop]
        return x, y, x2, y2

    def __getitem__(self, i):
        z = self.indices[i]
        x_z   = self._make_25d(z)
        x_zp1 = self._make_25d(z + 1)
        y_z   = self._get_mask(z)
        y_zp1 = self._get_mask(z + 1)

        x_z, y_z, x_zp1, y_zp1 = self._random_crop(x_z, y_z, x_zp1, y_zp1)

        x_z   = torch.from_numpy(x_z).float()     # HWC float [0,1]
        x_zp1 = torch.from_numpy(x_zp1).float()
        y_z   = torch.from_numpy(y_z).float()     # HW  float {0,1}
        y_zp1 = torch.from_numpy(y_zp1).float()

        return x_z, y_z, x_zp1, y_zp1
