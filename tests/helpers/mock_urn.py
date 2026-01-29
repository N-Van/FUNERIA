from pathlib import Path

import numpy as np
import tifffile


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


def delete_mock_urn(urn_path: Path, ground_truth_path: Path):
    urn_path.unlink(True)
    ground_truth_path.unlink(True)
