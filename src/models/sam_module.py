from typing import Literal, Optional, Tuple, cast

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from tqdm import tqdm
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

        print("init the sam model")
        self.detached_sam_model = DetachedSAM(sam_checkpoint)
        print("stop")
        self.test_loss = MeanMetric()
        self.worst_loss = MaxMetric()

        # loss function
        self.criterion = IntersectionOverUnion()

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        r"""Perform a forward pass through the model `self.net`.

        :param projections: A tensor of images (shape (Z, 3, H, W)), Z \ is along the projection
            axis)
        :return: A tensor with the 3D binary mask
        """
        print("=" * 5 + "😃 Image Batch's size: ", projections.size())
        print("=" * 5 + "😸 Image Batch's dtype: ", projections.dtype)
        depth, _, height, width = projections.shape
        mask_3D = torch.empty((depth, height, width), dtype=torch.bool, device=self.device)
        sam_inferrence_overrides = cast(
            Optional[IterableSimpleNamespace], self.hparams["sam_overrides"]
        )
        for z in tqdm(range(projections.shape[0]), desc="Segmenting projections", unit="projs"):
            frame = projections[z : z + 1]
            if (
                self.hparams["prompt_strategy"] is not None
                and self.hparams["prompt_strategy"] != "grid"
            ):
                # TODO: define a filter
                raise NotImplementedError(
                    "The custom prompt strategies to prelocate the bones are not yet implemented."
                )
            else:
                # TODO: define other hyperparametres
                # see https://docs.ultralytics.com/reference/models/sam/predict/#ultralytics.models.sam.predict.predictor.generate
                with torch.inference_mode():
                    results = self.detached_sam_model.sam_model(
                        frame,
                        **{
                            **dict(
                                imgsz=min(height, width),
                                # TODO: check if those two hyperparametres are actually set
                                # points_stride=self.hparams["points_stride"],
                                # points_batch_size=self.hparams["points_batch_size"]
                            ),
                            **dict(
                                sam_inferrence_overrides
                                if sam_inferrence_overrides is not None
                                else {}
                            ),
                        }
                    )[0]
                masks = results.masks  # shape (N, H, W)
                # N is the number of segments
                if masks is None:
                    raise Exception("Unexpected behavior: no mask found in the picture.")
                # TODO: use the scores (but with this line, not available in the API)
                # scores = results.scores
                # TODO: use the masks
                # boxes = results.boxes

                mask_3D[z] = cast(torch.Tensor, masks.data).sum(dim=0)
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
        similar_mask_3D = self.forward(y)
        # we expect the two masks to be similar
        # TODO: define the criterion
        loss = self.criterion(mask_3D, similar_mask_3D)
        # loss, preds, targets
        return loss, mask_3D, similar_mask_3D

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
        print("🥳 In test step")
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.worst_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/worst_loss", self.worst_loss, on_step=False, on_epoch=True, prog_bar=True)
        print("🎈 End test step")

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
