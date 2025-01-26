from pytest import fixture
import torch

IMAGE_SIZE = (1, 720, 720)
RESIZED_IMAGE_SIZE = (1, 448, 448)

@fixture
def test_image() -> torch.Tensor:
    return torch.rand(10, *IMAGE_SIZE)

@fixture
def resized_image() -> torch.Tensor:
    return torch.rand(1, *RESIZED_IMAGE_SIZE)
