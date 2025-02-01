from typing import List, Optional
from dataclasses import dataclass
from torch import nn, Tensor

@dataclass
class ConvLayerInputModel:

        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0

def create_multi_conv_layer_input(
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: Optional[List[int]] = None,
        paddings: Optional[List[int]] = None
) -> List[ConvLayerInputModel]:
        input = []
        for i in range(len(out_channels)):
                input.append(
                        ConvLayerInputModel(
                                in_channels=in_channels,
                                out_channels=out_channels[i],
                                kernel_size=kernel_sizes[i],
                                stride=strides[i] if strides else 1,
                                padding=paddings[i] if paddings else 0
                        )
                )
                in_channels = out_channels[i]
        return input

class ConvLayer(nn.Module):

        def __init__(self, input: ConvLayerInputModel):
                super(ConvLayer, self).__init__()
                self.conv2d = nn.Conv2d(input.in_channels, input.out_channels, input.kernel_size, input.stride, input.padding, bias=True)
                self.maxpool = nn.MaxPool2d(input.kernel_size, input.stride, input.padding)
                self.batchnorm = nn.BatchNorm2d(input.out_channels)
                self.activation = nn.LeakyReLU(0.1)

        def forward(self, x: Tensor, pooling: bool = True, normalize: bool = True):
                x = self.conv2d(x)
                if pooling:
                        x = self.maxpool(x)
                if normalize:
                        x = self.batchnorm(x)
                return self.activation(x)

class MultiConvLayer(nn.Module):

        def __init__(self, input: List[ConvLayerInputModel]):
                self.convolutions = []
                for model in input:
                        self.convolutions.append(ConvLayer(model))

        def forward(self, x: Tensor) -> Tensor:
                for i in range(len(self.convolutions) - 1):
                        x = self.convolutions[i].forward(x, pooling = False)
                return self.convolutions[-1].forward(x, pooling = True)
