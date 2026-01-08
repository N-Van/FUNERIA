"""Test tifffile fragment writing."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from lightning import LightningModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from src.models.types import OutputsWithLoss
from src.utils.imageproc.save_on_disk import SaveSegmentOnTheFly


class TestModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    @override
    def test_step(self, batch: Tensor) -> OutputsWithLoss:
        x, _ = batch
        loss = torch.empty((1, 1))
        return OutputsWithLoss(preds=x, loss=loss)


@pytest.mark.slow
def test_tifffile_save():
    """Generate a random segmentation and test the one-the-fly saving on disk.

    Write in a temporary file at `data/test_saved_tiff.tiff` position. This file is deleted at the
    end of the test.
    """
    data_dir = "data/"
    file = Path(data_dir) / "test_saved_tiff.tiff"
    callback = SaveSegmentOnTheFly(str(file))

    expected_shape = (50, 64, 64)
    batch_number = 10
    fragments = np.random.rand(*expected_shape) > 0.5
    dataloaders = DataLoader(
        TensorDataset(torch.tensor(fragments), torch.tensor(fragments)),
        batch_size=(expected_shape[0] // batch_number),
    )
    model = TestModule()
    trainer = Trainer(callbacks=[callback])
    trainer.test(model=model, dataloaders=dataloaders)

    assert file.exists(), "Expected output file does not exist."
    image = tifffile.imread(file)
    assert image.shape == fragments.shape
    assert np.all((image == 255) == fragments)
    file.unlink(missing_ok=True)
