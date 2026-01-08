from typing import TypedDict

from torch import Tensor


class OutputsWithLoss(TypedDict):
    """Return the predictions in addition to the loss value."""

    preds: Tensor
    loss: Tensor
