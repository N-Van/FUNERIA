import math
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, cast

import numpy as np
import tifffile as tiff
import torch
from lightning import LightningDataModule
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from src.utils.imageproc.image_25d import create_25d_image, create_25d_image_with_clahe
from src.utils.imageproc.resize import open_and_resize


class Compose:
    """Simple composer of functions."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x) -> Any:
        res = x
        for fn in self.transforms:
            res = fn(res)
        return res


class UrnDataset(Dataset[Tuple[NDArray[Any], NDArray[np.bool_]]]):
    """A Dataset which iteratively opens the 3D scans of each urn as a ndarray.

    TODO: load the correct segments
    """

    def __init__(self, target_segmentations_dir: None, tiff_pic_dir: Path):
        self.target_segmentations_dir = target_segmentations_dir  # TODO: use it
        self.tiff_files = [file for file in tiff_pic_dir.iterdir() if file.name.endswith("tiff")]

    @override
    def __getitem__(self, i: int):
        """Return the volume related to one Urn."""
        image = tiff.imread(self.tiff_files[i])
        correct_segments = np.empty_like(image, dtype=np.bool_)  # TODO:
        return image, correct_segments


class OneUrnDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """A Dataset which iterate over the projections of one urn along a defined axis.

    An item return a projection numpy picture of shape (S, S, 3) (H = W)
    """

    def __init__(
        self,
        image_path: str,
        slice_jump: int,
        correct_segments_path: str,
        transforms: Compose,
        rgb_transform: Callable,
        target_transforms: Compose,
        slice_image_size: Optional[int] = None,
        slicing_axis: Literal["x", "y", "z"] = "z",
    ) -> None:
        self.projection_axis = cast(
            Literal[0, 1, 2], {k: i for i, k in enumerate(("z", "y", "x"))}[slicing_axis]
        )
        self.slice_jump = slice_jump
        image, urn_slice_size = open_and_resize(
            Path(image_path), self.projection_axis, slice_image_size
        )
        self.image = rgb_transform(image)  # shape (Z, S, S, 3)
        self.image_slice_number = self.image.shape[self.projection_axis] // self.slice_jump
        # WARN: the full resolution for targets has not been kept
        # what prevents to show the resolution issues of the segmentation
        correct_segments, gd_slice_size = open_and_resize(
            Path(correct_segments_path), self.projection_axis, slice_image_size, boolify=True
        )
        if urn_slice_size != gd_slice_size:
            raise Exception(
                f"""The ground truth and urn tiff files must have the same slice side size. Received {gd_slice_size} and {urn_slice_size}

            See
            python src/utils.redim_urn.py --help
            """
            )
        self.correct_segments = correct_segments
        self.transforms = transforms
        self.target_transforms = target_transforms

    @override
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of projections fractioning one urn."""
        # TODO: why taking the first picture instead of the (irpn // pn) ones ?
        # TODO: interpolate
        idx_to_take = index * self.slice_jump
        projection_batch = np.take(self.image, indices=idx_to_take, axis=self.projection_axis)
        correct_segment_batch = np.take(
            self.correct_segments, indices=idx_to_take, axis=self.projection_axis
        )
        return self.transforms(projection_batch), self.target_transforms(correct_segment_batch)

    def __len__(self) -> int:
        """Return the number of projections."""
        return self.image_slice_number


def move_axis(source, destination):
    """Move the axes of the volume."""

    def move_axis__forward(x: np.ndarray):
        print("----------")
        print(x.shape)
        print("----------")
        return np.moveaxis(x, source, destination)

    return move_axis__forward


def normalize_channel():
    """Normalize the grayscale values between 0 and 1."""

    def normalize_channel__forward(x: np.ndarray):
        img = x
        mn, mx = float(img.min()), float(img.max())
        img = ((img - mn) / (mx - mn)) if mx > mn else np.zeros_like(img, dtype=np.float32)
        return img

    return normalize_channel__forward


def to_tensor(dtype: torch.dtype):
    """Transform the image of a projection into a tensor with values between 0 and 1."""

    def to_tensor__forward(x: np.ndarray):
        return torch.as_tensor(x, dtype=dtype)

    return to_tensor__forward


def to_25dimage(clahe: bool):
    """Convert a 3D volume into a 2D image by extracting the slice at `slice_idx` along the
    z-axis."""

    def to_25dimage__forward(volume: np.ndarray) -> np.ndarray:
        new_volume = np.empty((*volume.shape, 3), dtype=volume.dtype)
        if clahe:
            for slice_idx in range(volume.shape[0]):
                new_volume[slice_idx] = create_25d_image_with_clahe(
                    volume.astype(np.uint8), slice_idx, use_clahe=True
                )
        else:
            for slice_idx in range(volume.shape[0]):
                new_volume[slice_idx] = create_25d_image(volume, slice_idx, normalize=False)
        return new_volume

    return to_25dimage__forward


def grayscale_to_rgb():
    """Duplicate the single channel of the volume to return a grayscale rgb volume."""

    def grayscale_to_rgb__forward(volume: np.ndarray) -> np.ndarray:
        """Forward function.

        :param volume: shape (Z, S, S)
        :return: shape (Z, S, S, 3)
        """
        return volume[..., np.newaxis].repeat(3, -1)

    return grayscale_to_rgb__forward


class OneUrnDataModule(LightningDataModule):
    """`LightningDataModule` returning segments from a tiff file of one funeral urn.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        filename: str,
        ground_truth_filename: str,
        train_val_test_split: Tuple[int, int, int] = (0, 0, 1),
        slice_jump: int = 25,
        slice_image_size: Optional[int] = None,
        slicing_axis: Literal["x", "y", "z"] = "z",
        projection_batch_size: Optional[int] = None,  # projection frames per batch
        use_25d_image: Literal["clahe", True, False] = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `OneUrnDataModule`.

        :param filename: The file of the urn tiff image.
        :param ground_truth_filename: The file of the urn ground truth segmentation tiff image.
        :param train_val_test_split: The train, validation and test splits (number of slices). Defaults to `(0, 0, 1)`.
        :param slice_jump: The number of slices to be skipped along the slicing axis
        :param slice_image_size: The width and height value of a projection. If provided, a resized tiff image will be stored on disk. Else, it is expected the urn and ground truth tiff volumes to be isotropic.
        :param slicing_axis: The axis along which to slice the urn volume. Either `"x"`, `"y"` or `"z"`. Defaults to `"z"`.
        :param projection_batch_size: The number of projections per batch. Defaults to `None` to send all projections at once.
        :param use_25d_image: Whether to use 2.5D images with CLAHE preprocessing (`"clahe"`), without CLAHE (`True`).
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = Compose(
            [
                # adapt the shape to ultralytics' spec
                move_axis((2, 0, 1), (0, 1, 2)),
                normalize_channel(),
                to_tensor(torch.float32),
            ]
        )
        self.rgb_transform = (
            to_25dimage(clahe=True)
            if use_25d_image == "clahe"
            else (to_25dimage(clahe=False) if use_25d_image else grayscale_to_rgb())
        )
        self.target_transforms = Compose([to_tensor(torch.bool)])

        self.data_train: Optional[OneUrnDataset] = None
        self.data_val: Optional[OneUrnDataset] = None
        self.data_test: Optional[OneUrnDataset] = None

        # TODO: init the OneUrnDataset in the setup method
        # while preprocess the volume in the prepare_data method
        self._cached_data_test: Optional[OneUrnDataset] = None

        self.batch_size_per_device = projection_batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self._cached_data_test = OneUrnDataset(
            image_path=cast(str, self.hparams.get("filename")),
            slice_jump=cast(int, self.hparams.get("slice_jump")),
            slice_image_size=cast(Optional[int], self.hparams.get("slice_image_size")),
            correct_segments_path=cast(str, self.hparams.get("ground_truth_filename")),
            transforms=self.transforms,
            rgb_transform=self.rgb_transform,
            target_transforms=self.target_transforms,
            slicing_axis=cast(Literal["x", "y", "z"], self.hparams.get("slicing_axis")),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        batch_size = cast(Optional[int], self.hparams.get("projection_batch_size"))
        if batch_size is not None and self.trainer is not None:
            if batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_test:
            self.data_test = self._cached_data_test

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        raise NotImplementedError("No training for now.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        raise NotImplementedError("No training for now.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=cast(OneUrnDataset, self.data_test),
            batch_size=self.batch_size_per_device
            if self.batch_size_per_device is not None
            else len(cast(OneUrnDataset, self.data_test)),
            num_workers=self.hparams.get("num_workers", 0),
            pin_memory=self.hparams.get("pin_memory", False),
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
