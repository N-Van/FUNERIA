"""Callback to log in mlflow a graph over all the batch dimension."""

from typing import Any, List, cast

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from lightning import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from mlflow.tracking import MlflowClient
from torch.nn.modules import loss
from typing_extensions import override

from src.models.types import SegmentationForwardOutput, SegmentationLossNumpy


# TODO: adapt the slicing axis values to the slice jump
class MetricStackOverBatchAxis(Callback):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.metrics: dict[int, SegmentationLossNumpy] = {}

    @override
    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.metrics = {}
        return super().on_test_epoch_start(trainer, pl_module)

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        outputs = cast(SegmentationForwardOutput, outputs)
        self.metrics[batch_idx] = SegmentationLossNumpy(
            ground_truth_iou=outputs["loss"]["ground_truth_iou"].cpu().numpy(),
            pairwise_iou=outputs["loss"]["pairwise_iou"].cpu().numpy(),
        )
        return super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    @override
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        sorted_indices = sorted(self.metrics.keys())
        logger = cast(MLFlowLogger, trainer.logger)
        for loss_key in self.metrics[0]:
            complete_histogram = np.concatenate(
                [self.metrics[idx][loss_key] for idx in sorted_indices], axis=0
            )
            fig, ax = plt.subplots()
            ax.plot(complete_histogram)
            ax.set_xlabel("Slicing axis")
            ax.set_ylabel("IOU")
            ax.set_title(loss_key)
            cast(MlflowClient, logger.experiment).log_figure(
                run_id=cast(str, logger.run_id),
                figure=fig,
                artifact_file=f"plots/{loss_key}.png",
            )
            plt.close(fig)
        return super().on_test_epoch_end(trainer, pl_module)
