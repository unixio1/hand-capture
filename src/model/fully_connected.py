from torch import nn, Tensor

class ConnLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.3):
        super(ConnLayer, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x: Tensor, flatten: bool = True) -> Tensor:
        if flatten:
            x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return self.activation(x)

