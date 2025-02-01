from typing import List
from model.conv import ConvLayerInputModel, ConvLayer, MultiConvLayer, create_multi_conv_layer_input
from torch import Tensor
from torch import Size


kernel_size = 5 #Convolution filter of 5X5
padding = 0 # In case we want to mantain the size of the input, we add padding to the edges of the output (1 means, 1 extra pixel added)
stride = 1 #How the kernel moves, the step size (1 means it moves pixel by pixel applying convolutions)
in_channels = 1 #1 for grayscale, 3 for RGB
out_channels = 1
conv_layer_input = ConvLayerInputModel(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding
)
conv_layer = ConvLayer(conv_layer_input)

def test_conv_layer(resized_image: Tensor):
    expected_shape = _calculate_shape_after_convolution(resized_image, [out_channels], [kernel_size], pooling=False)
    conv_output = conv_layer.forward(resized_image, pooling=False)
    assert conv_output.shape == expected_shape

def test_conv_pooled_layer(resized_image: Tensor):
    expected_shape = _calculate_shape_after_convolution(resized_image, [out_channels], [kernel_size])
    output = conv_layer.forward(resized_image, pooling=True)
    assert output.shape == expected_shape 


def test_multi_conv_layer(resized_image: Tensor):
    kernel_sizes = [1, 3, 3]
    out_channels = [128, 256, 3]
    multi_conv_layer = MultiConvLayer(
        create_multi_conv_layer_input(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_sizes)
    )
    expected_shape = _calculate_shape_after_convolution(resized_image, out_channels, kernel_sizes)
    output = multi_conv_layer.forward(resized_image)
    assert output.shape == expected_shape

def _calculate_shape_after_convolution(image: Tensor, out_channels: List[int], kernel_sizes: List[int], pooling: bool = True) -> Size:
    assert len(out_channels) == len(kernel_sizes)
    image_shape = list(image.shape)
    new_size = image_shape[3]
    for i in range(len(out_channels)):
        new_size = (new_size - kernel_sizes[i]) + 1
    if pooling:
        new_size = (new_size - kernel_sizes[-1]) + 1 #This is regarding the last convolution involves a pooling, so we must calculate the ouptut of this operation
    return Size([image_shape[0], out_channels[-1], new_size, new_size])

