from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, cast

import numpy as np
import tifffile as tiff
from lightning import LightningDataModule
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing_extensions import override


class UrnDataset(Dataset[Tuple[NDArray[Any], None]]):
    """A Dataset which iteratively opens the 3D scans of each urn as a ndarray.

    TODO: load the correct segments
    """

    def __init__(self, target_segmentations_dir: None, tiff_pic_dir: Path):
        self.target_segmentations_dir = target_segmentations_dir  # TODO: use it
        self.tiff_files = [file for file in tiff_pic_dir.iterdir() if file.name.endswith("tiff")]

    @override
    def __getitem__(self, i: int):
        image = tiff.imread(self.tiff_files[i])
        correct_segments = None  # TODO:
        return image, correct_segments


class OneUrnDataset(Dataset[Tuple[NDArray[Any], None]]):
    """A Dataset which iterate over the projections of one urn along a defined axis."""

    def __init__(
        self,
        image: NDArray[Any],
        correct_segments: None,
        projection_axis: Literal["x", "y", "z"] = "z",
    ) -> None:
        self.projection_axis = {k: i for i, k in enumerate(("z", "y", "x"))}[projection_axis]
        self.image = image[..., np.newaxis].repeat(3, -1)
        self.correct_segments = correct_segments

    @override
    def __getitem__(self, index: int) -> Tuple[NDArray[Any], None]:
        return (np.take(self.image, indices=index, axis=self.projection_axis), None)

    def __len__(self) -> int:
        return self.image.shape[self.projection_axis]


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
        train_val_test_split: Tuple[int, int, int] = (0, 0, 1),
        batch_size: int = 25,  # frames per batch
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `OneUrnDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test splits (number of slices). Defaults to `(0, 0, 1)`.
        :param batch_size: The batch size. Defaults to `1`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # empty for now
        self.transforms = transforms.Compose([transforms.ToTensor])

        self.data_train: Optional[OneUrnDataset] = None
        self.data_val: Optional[OneUrnDataset] = None
        self.data_test: Optional[OneUrnDataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        batch_size = cast(int, self.hparams.get("batch_size"))
        if self.trainer is not None:
            if batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_test:
            image = tiff.imread(self.hparams.get("filename"))
            self.data_test = OneUrnDataset(image, None, "z")

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
            batch_size=self.batch_size_per_device,
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
