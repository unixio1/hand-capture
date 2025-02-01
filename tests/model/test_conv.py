from model.conv import ConvLayer, MultiConvLayer
from torch import Tensor
from torch import Size


kernel_size = 5 #Convolution filter of 5X5
padding = 0 # In case we want to mantain the size of the input, we add padding to the edges of the output (1 means, 1 extra pixel added)
stride = 1 #How the kernel moves, the step size (1 means it moves pixel by pixel applying convolutions)
in_channels = 1 #1 for grayscale, 3 for RGB
out_channels = 1
conv_layer = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)

def test_conv_layer(resized_image: Tensor):
    image_shape = list(resized_image.shape)
    new_image_size = ((resized_image.shape[3] - kernel_size + 2 * padding) // stride) + 1
    output_shape = Size([image_shape[0], out_channels, new_image_size, new_image_size])
    conv_output = conv_layer.forward(resized_image, pooling=False)
    assert conv_output.shape == output_shape

def test_conv_pooled_layer(resized_image: Tensor):
    image_shape = list(resized_image.shape)
    new_image_size = ((resized_image.shape[3] - kernel_size + 2 * padding) // stride) + 1
    conv_output_shape = [image_shape[0], out_channels, new_image_size, new_image_size]
    pool_output_size = ((conv_output_shape[3] - kernel_size + 2 * padding) // stride) + 1
    pool_output_shape = Size([*conv_output_shape[:2], pool_output_size, pool_output_size])
    output = conv_layer.forward(resized_image, pooling=True)
    assert output.shape == pool_output_shape 
    

def test_multi_conv_layer(resized_image: Tensor):
    image_shape = list(resized_image.shape)
    n_convolutions = 3
    kernel_sizes = [1, 3, 3]
    out_channels = [128, 256, 3]
    multi_conv_layer = MultiConvLayer(
        n_convolutions=n_convolutions,
        in_channels=1,
        out_channels=out_channels,
        kernel_sizes=kernel_sizes
    )
    new_size = image_shape[3]
    for i in range(n_convolutions):
        new_size = (new_size - kernel_sizes[i]) + 1
    new_size = (new_size - kernel_sizes[-1]) + 1 #This is regarding the last convolution involves a pooling, so we must calculate the ouptut of this operation
    pool_output_shape = Size([image_shape[0], out_channels[-1], new_size, new_size])
    output = multi_conv_layer.forward(resized_image)
    assert output.shape == pool_output_shape
