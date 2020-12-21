import torch
import torch.nn as nn
import torch.nn.functional as F

_PIECES = 12
_TENSOR_DIM = 16
_K = 2


def _depth_multiplier(depth):
    return _K**depth


class ChessConvNet(nn.Module):
    """
    This is very similar to the neural net used by @GeoHot in the TwitchChess example.
    """
    def __init__(self):
        super(ChessConvNet, self).__init__()

        self.conv1 = nn.Conv2d(_PIECES,
                               _TENSOR_DIM * _depth_multiplier(0),
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(0),
                               _TENSOR_DIM,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(0),
                               _TENSOR_DIM * _depth_multiplier(1),
                               kernel_size=3,
                               stride=2)

        self.b1 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(1),
                            _TENSOR_DIM * _depth_multiplier(1),
                            kernel_size=3,
                            padding=1)
        self.b2 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(1),
                            _TENSOR_DIM * _depth_multiplier(1),
                            kernel_size=3,
                            padding=1)
        self.b3 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(1),
                            _TENSOR_DIM * _depth_multiplier(2),
                            kernel_size=3,
                            stride=2)

        self.c1 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(2),
                            _TENSOR_DIM * _depth_multiplier(2),
                            kernel_size=2,
                            padding=1)
        self.c2 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(2),
                            _TENSOR_DIM * _depth_multiplier(2),
                            kernel_size=2,
                            padding=1)
        self.c3 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(2),
                            _TENSOR_DIM * _depth_multiplier(3),
                            kernel_size=2,
                            stride=2)

        self.d1 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(3),
                            _TENSOR_DIM * _depth_multiplier(3),
                            kernel_size=1)
        self.d2 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(3),
                            _TENSOR_DIM * _depth_multiplier(3),
                            kernel_size=1)
        self.d3 = nn.Conv2d(_TENSOR_DIM * _depth_multiplier(3),
                            _TENSOR_DIM * _depth_multiplier(3),
                            kernel_size=1)

        self.last = nn.Linear(_TENSOR_DIM * _depth_multiplier(3), 1)

    def forward(self, x):
        # Depth 0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Depth 1
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # Depth 2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        # Depth 3
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, _TENSOR_DIM * _depth_multiplier(3))
        x = self.last(x)

        # value output
        return torch.tanh(x)
