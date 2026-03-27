from pathlib import Path

import pytest
import torch
from ultralytics.utils import IterableSimpleNamespace

from src.models.sam_module import SAM3DModuleLinear

device = torch.device("cuda")
data_dir = "data/"
sam3D = SAM3DModuleLinear(
    str((Path(data_dir) / "sam_b.pt").resolve()), IterableSimpleNamespace(), points_stride=8
)


@pytest.mark.parametrize("stride_factor", [1, 2])
@pytest.mark.slow
def test_linear_sam_3D_pipeline(stride_factor: int):
    """Test if the forward call is correct for several image size."""
    stride = 32
    imgsz = stride_factor * stride
    Z, H, W = 3, imgsz, imgsz
    projection_examples = torch.rand((Z, 3, H, W), device=device)
    mask3D = sam3D(projection_examples)
    assert mask3D.dtype == torch.bool, "Incorrect output 3D mask dtype"
    assert mask3D.shape == (Z, H, W), "Incorrect expected output shape"
