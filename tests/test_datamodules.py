from pathlib import Path

import pytest
import torch

from src.data.one_urn_datamodule import OneUrnDataModule


@pytest.mark.parametrize(["slice_jump", "batch_size"], [(5, 1), (10, 1), (5, 5), (10, 5)])
def test_one_urn_datamodule(slice_jump: int, batch_size: int) -> None:
    """Tests `OneUrnDataModule` to verify that a test tiff file can be opened correctly, that the
    necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch
    sizes correctly match.

    :param slice_jump: Number of slices to jump while opening the tiff.
    :param batch_size: Number of slices to be loaded in each batch.
    """
    data_dir = "data/"

    mock_urn_filestem = "test_urn"
    mock_urn = Path(data_dir) / f"{mock_urn_filestem}.tiff"
    mock_urn_dim = 32
    slice_image_size = 32
    slicing_axis, slicing_axis_idx = "z", 0
    dm = OneUrnDataModule(
        filename=str(mock_urn),
        train_val_test_split=(0, 0, 1),
        slice_jump=slice_jump,
        slice_image_size=slice_image_size,
        projection_batch_size=batch_size,
        slicing_axis=slicing_axis,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(
        data_dir, f"{mock_urn_filestem}__{slice_image_size}_{slicing_axis_idx}.tiff"
    ).exists()

    dm.setup()
    # zero-shot for now
    # TODO:
    # assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
    assert dm.test_dataloader()
    assert dm.data_test

    total_slice_nb = len(dm.data_test)

    assert total_slice_nb == mock_urn_dim // slice_jump

    batch = next(iter(dm.test_dataloader()))
    x, y = batch
    assert len(x) == min(batch_size, total_slice_nb)
    assert len(y) == min(batch_size, total_slice_nb)
    assert x.dtype == torch.float32
    assert y.dtype == torch.bool
