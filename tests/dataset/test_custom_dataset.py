import torch
from dataset.custom_dataset import CustomDataset

def test_image_resizing(test_image: torch.Tensor):
    final_shape = [test_image.shape[0], 1, 448, 448]
    loader = CustomDataset(data=test_image, tensor_size=final_shape[2:])
    assert loader.data.shape == torch.Size(final_shape)

