"""Callback to save on-the-fly output segmentation fragments on the disk."""

from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import tifffile
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from src.models.types import SegmentationForwardOutput


class SaveSegmentOnTheFly(pl.Callback):
    """Save on disk the batch output.

    Attach this callback to a Module that return the output tensor in its test_step method.
    """

    def __init__(self, output_file: str) -> None:
        self.output_file = Path(output_file)
        self.tiff_writer: None | tifffile.TiffWriter = None

    @override
    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Open the TiffWriter on test epoch start."""
        self.tiff_writer = tifffile.TiffWriter(self.output_file)
        return super().on_test_epoch_start(trainer, pl_module)

    @override
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """At the end of a test, close the writer."""
        cast(tifffile.TiffWriter, self.tiff_writer).close()
        return super().on_test_epoch_end(trainer, pl_module)

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
        image = outputs["preds"].cpu().numpy() * 255
        writer = cast(tifffile.TiffWriter, self.tiff_writer)
        # add each frame sequentially
        for k in range(image.shape[0]):
            writer.write(image[k], contiguous=True)
        return super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
