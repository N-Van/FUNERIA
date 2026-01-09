from pathlib import Path
from typing import Any, Literal, Optional, Tuple, cast

import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy import ndimage


def binarize_image(image: NDArray[Any]) -> NDArray[Any]:
    """Return an image with only 0 or 255 value."""
    return (image > 255 / 2) * 255


def open_and_resize(
    tiff_path: Path,
    projection_axis: Literal[0, 1, 2],
    imgsz: Optional[int] = None,
    boolify=False,
) -> Tuple[NDArray[Any], int]:
    """Return a 3D volume with all axes excepted the projection one of dim imgsz.

    Store in the disk the resized tiff volume if newly produced.

    :param tiff_path: the path of the tiff raw image
    :param projection_axis: the axis that have not to be resized
    :param imgsz: the width and height of each slice. If not provided, then it is expected the
        source tiff image to have square slices.
    :return: the resized image as a ndarray and its slice side size
    """
    if imgsz is None:
        raw_volume = tifffile.imread(tiff_path)
        raw_shape = np.array(raw_volume.shape)
        slice_shape = tuple(raw_shape[i] for i in range(3) if i != projection_axis)
        if slice_shape[0] != slice_shape[1]:
            raise Exception(
                f"""The slices of the volume must be squares. Shape here is {slice_shape}.

            To create a valid, isotropic (cubic) volume for the urn. Please,
            use the src/utils.redim_urn.py script. See

            python src/utils.redim_urn.py --help
            """
            )
        return raw_volume, slice_shape[0]
    cached_image_path = tiff_path.parent / f"{tiff_path.stem}__{imgsz}_{projection_axis}.tiff"
    if cached_image_path.exists():
        return tifffile.imread(cached_image_path), imgsz
    raw_volume = tifffile.imread(tiff_path)
    raw_shape = np.array(raw_volume.shape)

    def keep_dim(axis_nb: int) -> int:
        return raw_volume.shape[axis_nb] if axis_nb == projection_axis else imgsz

    new_shape = np.array([keep_dim(i) for i in range(3)])
    zoom_factor = new_shape / raw_shape
    resized_volume = ndimage.zoom(raw_volume, zoom_factor, order=1)  # trilinear resizing
    if boolify:
        resized_volume = binarize_image(resized_volume)
    print("Resizing the volume...")
    tifffile.imwrite(cached_image_path, resized_volume)
    print("Resized image stored in", cached_image_path)
    return resized_volume, imgsz
