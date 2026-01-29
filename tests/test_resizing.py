from pathlib import Path
from typing import Literal

import pytest

from src.utils.imageproc.resize import open_and_resize
from tests.helpers.mock_urn import delete_mock_urn, generate_mock_urn_data

data_dir = "data/"


@pytest.mark.parametrize("proj_axis", [0, 1, 2])
@pytest.mark.parametrize("imgsz", [16, 32])
def test_image_resizing(proj_axis: Literal[0, 1, 2], imgsz: int):
    """Test the resizing of a tiff volume according to a projection axis."""
    mock_urn_filestem = "test_urn"
    mock_urn_gd_filestem = "test_urn_gd"
    slice_image_size = imgsz
    mock_urn, mock_urn_gd = generate_mock_urn_data(
        data_dir,
        mock_urn_filestem,
        mock_urn_gd_filestem,
        (slice_image_size, slice_image_size, slice_image_size),
    )
    img, slice_side_size = open_and_resize(mock_urn, proj_axis, imgsz)
    assert slice_side_size == imgsz
    expected_path = Path(data_dir) / f"test_urn__{imgsz}_{proj_axis}.tiff"
    for ax in range(3):
        if ax != proj_axis:
            assert (
                abs(img.shape[ax] - imgsz) <= 5
            ), f"Incorrect dim size for ax {ax}: got {img.shape[ax]}, expected {imgsz}"
    assert expected_path.exists()
    expected_path.unlink()
    delete_mock_urn(mock_urn, mock_urn_gd)
