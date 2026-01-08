from typing import Literal, Optional, Tuple, cast

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from tqdm import tqdm
from ultralytics.engine.results import Results
from ultralytics.models import SAM
from ultralytics.utils import IterableSimpleNamespace

from src.models.types import SegmentationForwardOutput, SegmentationLoss


class IntersectionOverUnion:
    """IOU for binary segmentation."""

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor):
        """Return the loss value with the IOU.

        :param image1: shape (Z, S, S)
        :param image2: shape (Z, S, S)
        :return: the per-slice loss value (shape (Z,))
        """
        return image1.logical_and(image2).sum(dim=(1, 2)) / image1.logical_or(image2).sum(
            dim=(1, 2)
        )


class DetachedSAM:
    r"""Wrapper for the LightningModule not to set the train mode of the \ ultralytics Model."""

    def __init__(self, sam_checkpoint: str) -> None:
        self._sam_model = SAM(sam_checkpoint)

    @property
    def sam_model(self) -> SAM:
        """Return the Ultralytics SAM model."""
        return self._sam_model


class SAM3DModuleLinear(LightningModule):
    """Predict 3D segments from a volume that can be crossed within slices.

    Given a tensor of slices of shape (Z, S, S), with Z the number of
    slices and S the size of the image (S will not be resized by
    ultralytics), return a segmented tensor of booleans of shape (Z, S, S).

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
        """Initialize a `SAM3DModuleLinear`.

        :param sam_checkpoint: Path to the pretrained weights of the SAM Model
        :param sam_overrides: Hyperparameters of SAM for the inference
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["sam_checkpoint"], logger=False)

        self.detached_sam_model = DetachedSAM(sam_checkpoint)
        self.test_gdloss = MeanMetric()
        self.worst_gdloss = MaxMetric()
        self.test_pwloss = MeanMetric()
        self.worst_pwloss = MaxMetric()

        # loss function
        self.criterion = IntersectionOverUnion()

    def generate_full_grid(self, stride: int, im_shape: tuple[int, int]):
        """Generate prompt points with a full grid."""
        h, w = im_shape
        ys = np.arange(stride // 2, h, stride)
        xs = np.arange(stride // 2, w, stride)
        pts = [(int(x), int(y)) for x in xs for y in ys]  # (x, y)
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

    def _filter_points_by_mask(self, points, urna_mask_np):
        # points: list[(x,y)]
        if urna_mask_np is None:
            return points
        return [(x, y) for (x, y) in points if urna_mask_np[y, x]]

    def _filter_masks(self, masks_bool: np.ndarray, urna_mask_np, min_area, max_area):
        # masks_bool: (N,H,W) bool
        if urna_mask_np is not None:
            masks_bool = np.logical_and(masks_bool, urna_mask_np[None, ...])

        areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
        keep = (areas >= min_area) & (areas <= max_area)
        return masks_bool[keep], areas, keep

    @torch.inference_mode()
    def infer_one_projection(
        self,
        frame: torch.Tensor,  # (1,3,H,W) torch
        urna_mask: Optional[np.ndarray] = None,  # (H,W) bool
        mode: str = "grid",  # "grid" | "points" | "auto"
        grid_stride: int = 32,
        points: Optional[list[tuple[int, int]]] = None,
        labels: Optional[list[int]] = None,
        points_batch_size: int = 25,
        imgsz: Optional[int] = None,
        min_area: int = 300,
        max_area_ratio: float = 0.05,
    ):
        _, _, H, W = frame.shape
        img_area = H * W
        max_area = int(max_area_ratio * img_area)

        if urna_mask is not None:
            urna_mask = urna_mask.astype(bool)
            if urna_mask.shape != (H, W):
                raise ValueError(f"urna_mask must be {(H,W)} got {urna_mask.shape}")
        model = self.detached_sam_model.sam_model
        overrides = self.hparams.get("sam_overrides", None)
        overrides = dict(overrides) if overrides is not None else {}
        imgsz = imgsz if imgsz is not None else min(H, W)

        # prompts
        if mode == "grid":
            pts_all, lbls_all = self.generate_full_grid(grid_stride, (H, W))  # (x,y)
            pts = self._filter_points_by_mask(pts_all, urna_mask)
            lbls = [1] * len(pts)
            if len(pts) == 0:
                return np.zeros((0, H, W), dtype=bool), {"n_raw": 0, "n_keep": 0, "n_points": 0}
        elif mode == "points":
            if points is None:
                raise ValueError("mode='points' needs points")
            pts = points
            lbls = labels if labels is not None else [1] * len(points)
        elif mode == "auto":
            pts, lbls = None, None
        else:
            raise ValueError("mode must be 'grid', 'points', or 'auto'")

        # run SAM in batches if points provided
        all_masks = []
        if pts is None:
            res = model.predict(source=frame, device=self.device, imgsz=imgsz, **overrides)[0]
            mask = self.get_mask(res)
            if mask is not None:
                all_masks.append(mask.detach().cpu().numpy().astype(bool))
        else:
            for i in range(0, len(pts), points_batch_size):
                res = model.predict(
                    source=frame,
                    points=pts[i : i + points_batch_size],
                    labels=(lbls[i : i + points_batch_size] if lbls is not None else None),
                    device=self.device,
                    imgsz=imgsz,
                    **overrides,
                )[0]

                if res.masks is not None:
                    m = cast(torch.Tensor, res.masks.data).detach().cpu().numpy()  # (N,H,W)
                    m = (m > 0.5).astype(bool)
                    all_masks.append(m)

        if len(all_masks) == 0:
            return np.zeros((0, H, W), dtype=bool), {
                "n_raw": 0,
                "n_keep": 0,
                "n_points": 0 if pts is None else len(pts),
            }

        masks = np.concatenate(
            [m if m.ndim == 3 else m[None, ...] for m in all_masks], axis=0
        )  # (N,H,W)
        masks_f, areas, keep = self._filter_masks(masks, urna_mask, min_area, max_area)

        info = {
            "n_raw": int(masks.shape[0]),
            "n_keep": int(masks_f.shape[0]),
            "n_points": 0 if pts is None else len(pts),
            "min_area": int(min_area),
            "max_area": int(max_area),
            "mode": mode,
        }
        return masks_f, info

    def forward(
        self, projections: torch.Tensor, urna_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        depth, _, H, W = projections.shape
        mask_3D = torch.zeros((depth, H, W), dtype=torch.bool, device=self.device)

        stride = int(self.hparams["points_stride"])
        bsz = int(self.hparams["points_batch_size"])

        for z in tqdm(range(depth), desc="Segmenting projections", unit="projs"):
            frame = projections[z : z + 1]  # (1,3,H,W)
            urna_mask_np = None
            if urna_masks is not None:
                urna_mask_np = urna_masks[z].detach().cpu().numpy().astype(bool)

            masks_f, info = self.infer_one_projection(
                frame,
                urna_mask=urna_mask_np,
                mode="grid",
                grid_stride=stride,
                points_batch_size=bsz,
            )
            if masks_f.shape[0] > 0:
                union = torch.from_numpy(masks_f.any(axis=0)).to(self.device)
                mask_3D[z].logical_or_(union)
        return mask_3D

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[SegmentationLoss, torch.Tensor, torch.Tensor]:
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

        # first loss: compare with the ground truth
        gd_loss = self.criterion(mask_3D, y)

        # second loss: pairwise-slice similarity
        # we expect the two masks to be similar
        second_similar_mask = mask_3D[1::2]
        first_mask = mask_3D[::2][: second_similar_mask.shape[0]]
        pairwise_loss = self.criterion(first_mask, second_similar_mask).repeat_interleave(
            2, dim=0
        )  # keep the same shape as the first loss

        # loss, preds, targets
        loss: SegmentationLoss = {
            "pairwise_iou": pairwise_loss,
            "ground_truth_iou": gd_loss,
        }
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
        return loss["ground_truth_iou"]

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

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> SegmentationForwardOutput:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_gdloss(loss["ground_truth_iou"].mean())
        self.worst_gdloss(loss["pairwise_iou"].mean())
        self.test_pwloss(loss["ground_truth_iou"].mean())
        self.worst_pwloss(loss["pairwise_iou"].mean())
        self.log("test/loss", self.test_gdloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/worst_loss", self.worst_gdloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pwloss", self.test_pwloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/worst_pwloss", self.test_pwloss, on_step=False, on_epoch=True, prog_bar=True
        )

        return SegmentationForwardOutput(preds=preds, loss=loss)

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
