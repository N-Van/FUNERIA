from typing import Tuple

import numpy as np
import pytest

from src.data.one_urn_datamodule import Compose, move_axis


@pytest.mark.parametrize("input_example", [0, 1, 5, -7, 89])
def test_compose_feature(input_example: int):
    """Tests the transform function composition.

    :param input_example: An input to be transformed.
    """
    f1 = lambda x: x + 2
    f2 = lambda x: x - 2

    identity_composition = Compose([f1, f2])
    assert identity_composition(input_example) == input_example


@pytest.mark.parametrize("shape", [(8, 5, 6), (28, 28, 3), (10, 32, 32, 3)])
def test_moveaxis_feature(shape: Tuple[int, ...]):
    """Test the move axis transform.

    :param shape: An example of ndarray shape
    """
    dim_nb = len(shape)
    transformer_fn = move_axis((dim_nb - 1, *range(dim_nb - 1)), (*range(dim_nb),))
    example_input = np.empty(shape)
    assert transformer_fn(example_input).shape == (shape[dim_nb - 1], *shape[:-1])
