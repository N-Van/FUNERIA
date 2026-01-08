from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from src.data.one_urn_datamodule import OneUrnDataModule


def generate_mock_urn_data(
    data_dir: str, filestem: str, gd_filestem: str, urn_shape: tuple[int, int, int]
) -> tuple[Path, Path]:
    """Write a mock urn in a file and a mock ground truth in another."""
    urn = np.random.randint(0, 256, urn_shape)
    urn_gd = (np.random.rand(*urn_shape) > 0.5) * 255
    urn_path = Path(data_dir) / f"{filestem}.tiff"
    urn_gd_path = Path(data_dir) / f"{gd_filestem}.tiff"
    tifffile.imwrite(urn_path, urn)
    tifffile.imwrite(urn_gd_path, urn_gd)
    return urn_path, urn_gd_path


@pytest.mark.slow
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
    mock_urn_gd_filestem = "test_urn_gd"
    slice_image_size = 32
    mock_urn_dim = 32
    mock_urn_shape = (mock_urn_dim, slice_image_size, slice_image_size)
    mock_urn, mock_urn_gd = generate_mock_urn_data(
        data_dir, mock_urn_filestem, mock_urn_gd_filestem, mock_urn_shape
    )
    slicing_axis, slicing_axis_idx = "z", 0
    dm = OneUrnDataModule(
        filename=str(mock_urn),
        ground_truth_filename=str(mock_urn_gd),
        train_val_test_split=(0, 0, 1),
        slice_jump=slice_jump,
        slice_image_size=slice_image_size,
        projection_batch_size=batch_size,
        slicing_axis=slicing_axis,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    resized_urn_file = Path(
        data_dir, f"{mock_urn_filestem}__{slice_image_size}_{slicing_axis_idx}.tiff"
    )
    resized_gd_file = Path(
        data_dir, f"{mock_urn_gd_filestem}__{slice_image_size}_{slicing_axis_idx}.tiff"
    )
    assert resized_urn_file.exists()
    resized_urn_file.unlink(True)
    assert resized_gd_file.exists()
    resized_gd_file.unlink(True)

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
    assert x.shape[1:] == (3, slice_image_size, slice_image_size)
    assert len(y) == min(batch_size, total_slice_nb)
    assert y.shape[1:] == (slice_image_size, slice_image_size)
    assert x.dtype == torch.float32
    assert y.dtype == torch.bool

    mock_urn.unlink(True)
    mock_urn_gd.unlink(True)
