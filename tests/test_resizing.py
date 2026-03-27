from pathlib import Path
from typing import Literal

import pytest

from src.utils.imageproc.resize import open_and_resize

data_dir = "data/"


@pytest.mark.parametrize("proj_axis", [0, 1, 2])
@pytest.mark.parametrize("imgsz", [16, 32])
def test_image_resizing(proj_axis: Literal[0, 1, 2], imgsz: int):
    """Test the resizing of a tiff volume according to a projection axis."""
    mock_urn = Path(data_dir) / "test_urn.tiff"
    img = open_and_resize(mock_urn, proj_axis, imgsz)
    expected_path = Path(data_dir) / f"test_urn__{imgsz}_{proj_axis}.tiff"
    for ax in range(3):
        if ax != proj_axis:
            assert (
                abs(img.shape[ax] - imgsz) <= 5
            ), f"Incorrect dim size for ax {ax}: got {img.shape[ax]}, expected {imgsz}"
    assert expected_path.exists()
    expected_path.unlink()
