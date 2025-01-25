from typing import Optional, List
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class CustomDataset(Dataset):
    """"""
    def __init__(self, data: Optional[torch.Tensor], tensor_size: List[int] = [448, 448]):
        if data is not None:
            self.data = self._reshape(data, tensor_size)

    def _reshape(self, data: torch.Tensor, tensor_size: List[int]) -> torch.Tensor:
        """Reshapes the tensor into the given sizes"""
        if len(data.shape) == 3:
            data = data.unsqueeze(1)
        resized = F.interpolate(data, size=tensor_size)
        return resized

