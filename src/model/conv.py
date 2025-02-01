from typing import List
from torch import nn, Tensor

class ConvLayer(nn.Module):

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
                super(ConvLayer, self).__init__()
                self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
                self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
                self.batchnorm = nn.BatchNorm2d(out_channels)
                self.activation = nn.LeakyReLU(0.1)
        
        def forward(self, x: Tensor, pooling: bool = True, normalize: bool = True):
                x = self.conv2d(x)
                if pooling:
                        x = self.maxpool(x)
                if normalize:
                        x = self.batchnorm(x)
                return self.activation(x)

class MultiConvLayer(nn.Module):

        def __init__(self, n_convolutions: int, in_channels: int, out_channels: List[int], kernel_sizes: List[int]):
                self.convolutions = []
                for i in range(n_convolutions):
                        self.convolutions.append(ConvLayer(in_channels, out_channels[i], kernel_sizes[i]))
                        in_channels = out_channels[i]

        def forward(self, x: Tensor) -> Tensor:
                for i in range(len(self.convolutions) - 1):
                        x = self.convolutions[i].forward(x, pooling = False)
                return self.convolutions[-1].forward(x, pooling = True)
