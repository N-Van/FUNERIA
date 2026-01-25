"""Callback to save on-the-fly output segmentation fragments on the disk (as a Z-stack mask)."""

from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import tifffile
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from src.models.types import SegmentationForwardOutput


class SaveSegmentOnTheFly(pl.Callback):
    """Save on disk the batch output, slice by slice, as a grayscale mask (Z, H, W)."""

    def __init__(self, output_file: str) -> None:
        self.output_file = Path(output_file)
        self.tiff_writer: tifffile.TiffWriter | None = None
    def _close_writer(self) -> None:
        if self.tiff_writer is not None:
            try:
                self.tiff_writer.close()
            finally:
                self.tiff_writer = None

    @override
    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.tiff_writer = tifffile.TiffWriter(self.output_file, bigtiff=True)

    @override
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._close_writer()

    @override
    def on_exception(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException
    ) -> None:
        #  si crash, on ferme proprement sinon TIFF invalide
        self._close_writer()

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """At each test step, save on the fly the new segmented fragment."""
        outputs = cast(SegmentationForwardOutput, outputs)

        #  (B, H, W) bool ou float [0..1]
        pred = outputs["preds"].detach().cpu().numpy()

        if pred.dtype == np.bool_:
            mask = pred
        else:
            mask = pred > 0.5

        mask_u8 = mask.astype(np.uint8) * 255  # (B,H,W)

        writer = cast(tifffile.TiffWriter, self.tiff_writer)
        for k in range(mask_u8.shape[0]):
            writer.write(mask_u8[k], photometric="minisblack", contiguous=True)
