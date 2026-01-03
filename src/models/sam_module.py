from typing import Literal, Optional, Tuple, cast

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from tqdm import tqdm
from ultralytics.engine.results import Results
from ultralytics.models import SAM
from ultralytics.utils import IterableSimpleNamespace


class IntersectionOverUnion:
    """IOU for binary segmentation."""

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor):
        """Return the loss value with the IOU."""
        return image1.logical_and(image2).sum() / image1.logical_or(image2).sum()


class DetachedSAM:
    r"""Wrapper for the LightningModule not to set the train mode of the \ ultralytics Model."""

    def __init__(self, sam_checkpoint: str) -> None:
        self._sam_model = SAM(sam_checkpoint)

    @property
    def sam_model(self) -> SAM:
        """Return the Ultralytics SAM model."""
        return self._sam_model


class SAM3DModuleLinear(LightningModule):
    """Predict 3D segments from a volume that can be crossed within projections.

    Give a tensor of projections of shape (Z, S, S), with Z the number of
    projections and S the size of the image (S will not be resized by
    ultralytics).

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        sam_checkpoint: str,
        sam_overrides: Optional[IterableSimpleNamespace] = None,
        points_stride: int = 32,
        points_batch_size: int = 25,
        # TODO: define custom prompt strategy
        prompt_strategy: Optional[Literal["grid"]] = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param sam_checkpoint: Path to the pretrained weights of the SAM Model
        :param sam_overrides: Hyperparameters of SAM for the inference
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["sam_checkpoint"], logger=False)

        self.detached_sam_model = DetachedSAM(sam_checkpoint)
        self.test_loss = MeanMetric()
        self.worst_loss = MaxMetric()

        # loss function
        self.criterion = IntersectionOverUnion()

    def generate_full_grid(self, stride: int, im_shape: tuple[int, int]):
        """Generate prompt points with a full grid."""
        h, w = im_shape
        ys = np.arange(stride // 2, h, stride)
        xs = np.arange(stride // 2, w, stride)
        pts = [(int(y), int(x)) for x in xs for y in ys]  # (x, y)
        label = 1
        lbls = [label] * len(pts)
        return pts, lbls

    @staticmethod
    def get_mask(results: Results):
        """Return the mask from the output of the model.

        :param results: the output
        :param initial_projection: the initial input (shape
        """
        binary_treshold = 0.5
        masks = results.masks  # shape [N, H, W]
        if masks is None:
            return None
        return (cast(torch.Tensor, masks.data) > binary_treshold).sum(dim=0) > 0

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        r"""Perform a forward pass through the model `self.net`.

        :param projections: A tensor of images (shape (Z, 3, H, W)), Z is along the projection
            axis)
        :return: A tensor with the 3D binary mask
        """
        depth, _, height, width = projections.shape
        mask_3D = torch.zeros((depth, height, width), dtype=torch.bool, device=self.device)
        sam_inferrence_overrides = cast(
            Optional[IterableSimpleNamespace], self.hparams["sam_overrides"]
        )
        points_stride = cast(int, self.hparams["points_stride"])
        points_batch_size = cast(int, self.hparams["points_batch_size"])
        points, labels = self.generate_full_grid(points_stride, (height, width))
        model = self.detached_sam_model.sam_model
        for z in tqdm(range(projections.shape[0]), desc="Segmenting projections", unit="projs"):
            frame = projections[z : z + 1]
            # TODO: generate here the filtered points and labels
            for i in tqdm(
                range(0, len(points), points_batch_size),
                desc="Processed prompt point batches",
                unit="batches",
            ):
                results = model.predict(
                    source=frame,
                    points=points[i : i + points_batch_size],
                    labels=labels[i : i + points_batch_size],
                    device=self.device,
                    imgsz=min(height, width),
                    **dict(
                        sam_inferrence_overrides if sam_inferrence_overrides is not None else {}
                    ),
                )[0]
                mask = self.get_mask(results)
                if mask is not None:
                    mask_3D[z].logical_or_(mask)
                # TODO: use the scores (but with this line, not available in the API)
                # scores = results.scores
                # TODO: use the masks
                # boxes = results.boxes
        # TODO: use the scores and merge the boxes
        return mask_3D

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

                :param batch: A batch of data (a tuple) containing as input tensor \
        the even-indexed frames along the projection axis, and the other even-indexed \
        frames as target tensor

                :return: A tuple containing (in order):
                    - A tensor of losses
                    - A tensor of predictions.
                    - A tensor of target labels.
        """
        x, y = batch
        mask_3D = self.forward(x)
        # we expect the two masks to be similar
        second_similar_mask = mask_3D[1::2]
        first_mask = mask_3D[::2][: second_similar_mask.shape[0]]
        loss = self.criterion(first_mask, second_similar_mask)
        # loss, preds, targets
        # return the full mask for now
        return loss, mask_3D, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.worst_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/worst_loss", self.worst_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass
