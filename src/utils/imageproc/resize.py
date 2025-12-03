from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy import ndimage


def open_and_resize(
    tiff_path: Path, projection_axis: Literal[0, 1, 2], imgsz: int
) -> NDArray[Any]:
    """Return a 3D volume with all axes excepted the projection one of dim imgsz.

    Store in the disk the resized tiff volume if newly produced.

    :param tiff_path: the path of the tiff raw image
    :param projection_axis: the axis that have not to be resized
    :param imgsz: the width and height of each projection
    :return: the resized image as a ndarray
    """
    cached_image_path = tiff_path.parent / f"{tiff_path.stem}__{imgsz}_{projection_axis}.tiff"
    if cached_image_path.exists():
        return tifffile.imread(cached_image_path)
    raw_volume = tifffile.imread(tiff_path)
    raw_shape = np.array(raw_volume.shape)

    def keep_dim(axis_nb: int) -> int:
        return raw_volume.shape[axis_nb] if axis_nb == projection_axis else imgsz

    new_shape = np.array([keep_dim(i) for i in range(3)])
    zoom_factor = new_shape / raw_shape
    resized_volume = ndimage.zoom(raw_volume, zoom_factor, order=1)  # trilinear resizing
    print("Resizing the volume...")
    tifffile.imwrite(cached_image_path, resized_volume)
    print("Resized image stored in", cached_image_path)
    return resized_volume
