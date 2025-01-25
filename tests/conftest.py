from pytest import fixture
import torch

IMAGE_SIZE = (720, 720)
@fixture
def test_image() -> torch.Tensor:
    return torch.rand(10, *IMAGE_SIZE)
