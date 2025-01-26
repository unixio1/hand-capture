from model.conv import ConvLayer
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
    output_shape = Size([*image_shape[:2], new_image_size, new_image_size])
    conv_output = conv_layer.forward(resized_image, pooling=False)
    assert conv_output.shape == output_shape

def test_conv_pooled_layer(resized_image: Tensor):
    image_shape = list(resized_image.shape)
    new_image_size = ((resized_image.shape[3] - kernel_size + 2 * padding) // stride) + 1
    conv_output_shape = [*image_shape[:2], new_image_size, new_image_size]
    pool_output_size = ((conv_output_shape[3] - kernel_size + 2 * padding) // stride) + 1
    pool_output_shape = Size([*conv_output_shape[:2], pool_output_size, pool_output_size])
    output = conv_layer.forward(resized_image, pooling=True)
    assert output.shape == pool_output_shape 
    



