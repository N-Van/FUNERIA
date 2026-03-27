from typing import TypedDict

from numpy.typing import NDArray
from torch import Tensor


class SegmentationLoss(TypedDict):
    """Ground truth loss and pairwise loss."""

    ground_truth_iou: Tensor
    pairwise_iou: Tensor


class SegmentationLossNumpy(TypedDict):
    """Numpy version."""

    ground_truth_iou: NDArray
    pairwise_iou: NDArray


class SegmentationForwardOutput(TypedDict):
    """Return the predictions in addition to the loss value."""

    preds: Tensor
    loss: SegmentationLoss
