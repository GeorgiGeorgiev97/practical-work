"""
DenseNet components for M-heads implementation
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """Bottleneck layer for DenseNet"""

    def __init__(self, in_channels, growth_rate, dropout_rate=0.0, bottleneck_rate=4, padding=1):
        """

        Initialize Bottleneck layer.

        Args:
            in_channels (int): Number of input channels
            growth_rate (int): Growth rate (k in the paper)
            dropout_rate (float): Dropout rate
            bottleneck_rate (int): Bottleneck rate for 1x1 conv
            padding (int): Padding for 3x3 conv
        """
        super(Bottleneck, self).__init__()

        inter_channels = bottleneck_rate * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=padding, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))

        if self.dropout is not None:
            out = self.dropout(out)

        return torch.cat([x, out], 1)


class SingleLayer(nn.Module):
    """Single convolutional layer for DenseNet (non-bottleneck)."""

    def __init__(self, in_channels, growth_rate, dropout_rate=0.0, padding=1):
        """
        Initialize SingleLayer.

        Args:
            in_channels (int): Number of input channels
            growth_rate (int): Growth rate (k in the paper)
            dropout_rate (float): Dropout rate
            padding (int): Padding for 3x3 conv
        """
        super(SingleLayer, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=padding, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = self.conv(self.act(self.bn(x)))

        if self.dropout is not None:
            out = self.dropout(out)

        return torch.cat([x, out], 1)


class Transition(nn.Module):
    """Transition layer between dense blocks."""

    def __init__(self, in_channels, out_channels, padding=1):
        """
        Initialize Transition layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            padding (int): Padding parameter (not used for pooling, kept for API compatibility)
        """
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Don't use padding for average pooling - it should always be 0
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.conv(self.act(self.bn(x)))
        out = self.pool(out)
        return out


def make_dense_block(in_channels, growth_rate, n_layers, bottleneck=True,
                     dropout_rate=0.0, bottleneck_rate=4, padding=1):
    """
    Create a dense block.

    Args:
        in_channels (int): Number of input channels
        growth_rate (int): Growth rate (k in the paper)
        n_layers (int): Number of layers in the block
        bottleneck (bool): Whether to use bottleneck layers
        dropout_rate (float): Dropout rate
        bottleneck_rate (int): Bottleneck rate for 1x1 conv
        padding (int): Padding for convolutions

    Returns:
        nn.Sequential: Dense block
    """
    layers = []

    for i in range(n_layers):
        if bottleneck:
            layer = Bottleneck(
                in_channels + i * growth_rate,
                growth_rate,
                dropout_rate,
                bottleneck_rate,
                padding
            )
        else:
            layer = SingleLayer(
                in_channels + i * growth_rate,
                growth_rate,
                dropout_rate,
                padding
            )
        layers.append(layer)

    return nn.Sequential(*layers)
