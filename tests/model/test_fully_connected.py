from model.fully_connected import ConnLayer
from torch import Tensor, Size

def test_fully_connected_layer(resized_image: Tensor):
    input_shape = resized_image.shape
    number_of_pixels = input_shape[1] * input_shape[2] * input_shape[3]
    in_features = number_of_pixels
    out_features = 120
    fully_connected_layer = ConnLayer(in_features, out_features)
    output_tensor = fully_connected_layer.forward(resized_image, flatten = True)
    expected_shape = Size([input_shape[0], out_features])
    assert expected_shape == output_tensor.shape
