"""
File contains user defined layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Implements swish activation function : https://arxiv.org/abs/1710.05941
    If Beta = 1 (default), swish acts as SiLU
    """

    def __init__(self, beta: float = 1.0) -> None:
        """Inits swish activation with a given Beta

        Args:
            beta (float, optional): Beta of swish.
            beta = 1 turns swish into Silu activation. Defaults to 1.0.
        """
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swish forward function"""
        return x * F.sigmoid(x * float(self.beta))


class PixelShuffleUpsample(nn.Module):
    """
    Adapted from Imagen-pytorch by lucidrains
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Init pixelshuffle upscaler

        Args:
            input_dim (int): number of input dims
            output_dim (int): Number of desired output dims
        """
        super().__init__()
        conv = nn.Conv2d(input_dim, output_dim * 4, 1)

        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))

        self.init_conv_(conv)

    def init_conv_(self, conv: nn.Module) -> None:
        """Inits conv weights"""
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = conv_weight.repeat(4, 1, 1, 1)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""
        return self.net(x)


class ResNetBlock(nn.Module):
    """Constructs a Resnet Block
    Norm->activation->conv
            +
    Norm->activation->conv
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 32,
        skip_connection_scale: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        """Creates a residual block

        Args:
            channels (int): Number of Input channels to the block
            num_groups (int, optional): Number of groups to be used for Groupnorm
                (batchnorm if 0). Defaults to 32.
            skip_connection_scale (float, optional): Value by with to scale
                the skip connection. Defaults to 1.
            beta (float, optional): beta for swish activation function. Defaults to 1.0.
        """
        super(ResNetBlock, self).__init__()
        norm_layer = (
            nn.GroupNorm(num_groups, channels)
            if num_groups != 0
            else nn.BatchNorm2d(channels)
        )

        self.main_path = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            Swish(beta),
            nn.Conv2d(channels, channels, 3, 1, 1),
            norm_layer,
            Swish(beta),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        self.skip_conn_scale = skip_connection_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""
        x_main = self.main_path(x)
        return x_main + x * self.skip_conn_scale
