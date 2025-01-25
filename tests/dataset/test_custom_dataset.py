import torch
from dataset.custom_dataset import CustomDataset

def test_image_resizing(test_image: torch.Tensor):
    loader = CustomDataset(data=test_image, tensor_size=[448, 448])
    assert loader.data.shape == torch.Size([test_image.shape[0], 1, 448, 448])

