#!/usr/bin/env python3
import os
import gc
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


def diagnose_tiff(path: str):
    with tifffile.TiffFile(path) as tif:
        pages = len(tif.pages)
        series_n = len(tif.series)
        series_shapes = [s.shape for s in tif.series]
        series_dtypes = [str(s.dtype) for s in tif.series]
    return pages, series_n, series_shapes, series_dtypes


def ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    dr = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    return 1.0 if dr == 0 else float(ssim(a, b, data_range=dr))


def ssim_curve_axis(vol: np.ndarray, axis: int, desc: str) -> np.ndarray:
    n = vol.shape[axis]
    out = np.empty(n - 1, dtype=np.float32)
    for i in tqdm(range(n - 1), desc=desc):
        A = np.take(vol, i, axis=axis)
        B = np.take(vol, i + 1, axis=axis)
        out[i] = ssim_pair(A, B)
        del A, B
        gc.collect()
    return out


def ssim_curves_2d_bands(img: np.ndarray, band: int):
    img = img.astype(np.float32, copy=False)
    H, W = img.shape

    # Y: bandes horizontales
    ssim_y = np.empty(max(0, H - band), dtype=np.float32)
    for y in tqdm(range(max(0, H - band)), desc=f"SSIM axe Y (bandes={band})"):
        A = img[y:y + band, :]
        B = img[y + 1:y + 1 + band, :]
        ssim_y[y] = ssim_pair(A, B)
        del A, B
        gc.collect()

    # X: bandes verticales
    ssim_x = np.empty(max(0, W - band), dtype=np.float32)
    for x in tqdm(range(max(0, W - band)), desc=f"SSIM axe X (bandes={band})"):
        A = img[:, x:x + band]
        B = img[:, x + 1:x + 1 + band]
        ssim_x[x] = ssim_pair(A, B)
        del A, B
        gc.collect()

    return ssim_x, ssim_y


def plot_and_save(curve: np.ndarray, out_png: str, title: str, xlabel: str):
    plt.figure(figsize=(9, 4))
    plt.plot(curve)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("SSIM")
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Explore SSIM curves along axes for TIFF 2D/3D")
    ap.add_argument("--path", required=True, help="Path to .tif/.tiff file")
    ap.add_argument("--outdir", default="ssim_out", help="Output directory")
    ap.add_argument("--prefer-memmap", action="store_true", help="Try tifffile.memmap first")
    ap.add_argument("--band", type=int, default=16, help="Band thickness for 2D fallback (X/Y curves)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Diagnose ---
    pages, series_n, series_shapes, series_dtypes = diagnose_tiff(args.path)
    print(f"[TIFF] path={args.path}")
    print(f"[TIFF] pages={pages}, series={series_n}")
    for i, (sh, dt) in enumerate(zip(series_shapes, series_dtypes)):
        print(f"[TIFF] series[{i}] shape={sh}, dtype={dt}")

    # --- Load (robust) ---
    V = None
    load_mode = None

    if args.prefer_memmap:
        try:
            V = tifffile.memmap(args.path)
            load_mode = "memmap"
        except Exception as e:
            print(f"[WARN] memmap failed: {repr(e)}")

    if V is None:
        # fallback: normal read
        V = tifffile.imread(args.path)
        load_mode = "imread"

    print(f"[LOAD] mode={load_mode} shape={V.shape} ndim={V.ndim} dtype={V.dtype}")

    
    if V.ndim == 3:
        ssim_z = ssim_curve_axis(V, axis=0, desc="SSIM axe Z")
        ssim_y = ssim_curve_axis(V, axis=1, desc="SSIM axe Y")
        ssim_x = ssim_curve_axis(V, axis=2, desc="SSIM axe X")

        np.save(os.path.join(args.outdir, "ssim_z.npy"), ssim_z)
        np.save(os.path.join(args.outdir, "ssim_y.npy"), ssim_y)
        np.save(os.path.join(args.outdir, "ssim_x.npy"), ssim_x)

        plot_and_save(ssim_z, os.path.join(args.outdir, "ssim_z.png"),
                      "SSIM entre tranches consécutives - axe Z", "index z (z vs z+1)")
        plot_and_save(ssim_y, os.path.join(args.outdir, "ssim_y.png"),
                      "SSIM entre coupes consécutives - axe Y", "index y (y vs y+1)")
        plot_and_save(ssim_x, os.path.join(args.outdir, "ssim_x.png"),
                      "SSIM entre coupes consécutives - axe X", "index x (x vs x+1)")

        print(f"[DONE] Saved .npy and .png in: {args.outdir}")

    elif V.ndim == 2:
        ssim_x, ssim_y = ssim_curves_2d_bands(V, band=args.band)

        np.save(os.path.join(args.outdir, "ssim_x_2d.npy"), ssim_x)
        np.save(os.path.join(args.outdir, "ssim_y_2d.npy"), ssim_y)

        plot_and_save(ssim_y, os.path.join(args.outdir, "ssim_y_2d.png"),
                      f"SSIM bandes consécutives — axe Y (band={args.band})", "index y")
        plot_and_save(ssim_x, os.path.join(args.outdir, "ssim_x_2d.png"),
                      f"SSIM bandes consécutives — axe X (band={args.band})", "index x")

        print(f"[DONE] 2D TIFF detected → computed X/Y banded SSIM only. Output: {args.outdir}")
        print("[NOTE] Pour avoir Z/Y/X, il faut un vrai stack 3D (multi-pages).")

    else:
        raise ValueError(f"Unsupported ndim={V.ndim}. Expected 2D or 3D.")

    # cleanup
    del V
    gc.collect()


if __name__ == "__main__":
    main()
