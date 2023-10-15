"""
File contains definitions of Base unet layers and Unet model definition
"""
import torch
import torch.nn as nn

from plusultra.models.base import PixelShuffleUpsample, ResNetBlock, Swish


class DownBlock(nn.Module):
    """Definition of Unet Downsample block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        numResNetBlocks: int,
        num_groups: int = 32,
        skip_conn_scale: float = 1.0,
        beta: float = 1.0,
        downsample_conv: bool = False,
    ) -> None:
        """Initializes a Downsample block followed by a resnet block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            numResNetBlocks (int): Number of resnet blocks that follow downsample block
            num_groups (int, optional): number of groups in groupnorm. Defaults to 32.
            skip_conn_scale (float, optional): Scle of skipconnections in resnetblocks.
                Defaults to 1.0.
            beta (float, optional): Beta for swish. Defaults to 1.0.
            use_conv (bool, optional): Use convolution to downsample or do maxpooling.
                Defaults to False.
        """
        super(DownBlock, self).__init__()

        conv_stride = 2 if downsample_conv else 1
        self.down_sample_conv = nn.Conv2d(in_channels, out_channels, 3, conv_stride, 1)
        if not downsample_conv:
            self.max_pool = nn.AvgPool2d(2, 2)
            self.downsample = nn.Sequential(self.max_pool, self.down_sample_conv)
        else:
            self.downsample = self.down_sample_conv
        self.resblocks = nn.ModuleList(
            [
                ResNetBlock(out_channels, num_groups, skip_conn_scale, beta)
                for i in range(numResNetBlocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""
        x = self.downsample(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class UpBlock(nn.Module):
    """Definition of Unet Upscaling block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        numResNetBlocks: int,
        num_groups: int = 32,
        skip_conn_scale: float = 1.0,
        beta: float = 1.0,
        use_pixelshuffle: float = False,
    ) -> None:
        """Initialize an upsample block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output Channels
            numResNetBlocks (int): Number of Resenet blocks
            num_groups (int, optional): Number of groups for groupnorm. Defaults to 32.
            skip_conn_scale (int, optional): Scale of resnet skip connections.
                Defaults to 1.
            beta (float, optional): beta for swish. Defaults to 1.0.
            use_pixelshuffle (bool, optional): Whether to use pixelshuffle
                or nearest neighbour interpolation. Defaults to False.
        """
        super(UpBlock, self).__init__()
        if use_pixelshuffle:
            self.upsample = nn.Sequential(
                PixelShuffleUpsample(in_channels, in_channels),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )
        self.resblocks = nn.ModuleList(
            [
                ResNetBlock(in_channels, num_groups, skip_conn_scale, beta)
                for i in range(numResNetBlocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.upsample(x)

        return x


class Unet(nn.Module):
    """Unet Model Definition"""

    def __init__(
        self,
        base_channels: int = 32,
        num_resblocks: list[int] = [1, 1, 1, 1],
        beta: float = 1.0,
        num_groups: int = 32,
        skip_conn_scale: float = 1.0,
        downsample_conv: bool = False,
        use_pixelshuffle: bool = False,
        upscale: bool = True,
    ) -> None:
        """Initialize a Unet Model

        Args:
            base_channels (int, optional): Number of Channels in first block.
                Defaults to 32. Number of channels double every block
            num_resblocks (list[int], optional): Number of resblocks to use in each block.
                Defaults to [1, 1, 1, 1].
            beta (float, optional): beta of swish. Defaults to 1.0.
            num_groups (int, optional): Number of groups for groupnorm. Defaults to 32.
            skip_conn_scale (float, optional): Scaling factor for skip connecions.
                Defaults to 1.0.
            downsample_conv (bool, optional): Whether to use convolution to downsample
                or averagepool. Defaults to False.
            use_pixelshuffle (bool, optional): whether to use pixelshuffle for upsample or
                nearest neighbout. Defaults to False.
            upscale (bool, optional): Whether to add an aditional upscaler.
                 Defaults to True. increases spacial resolution by 2x
        """
        super(Unet, self).__init__()

        input_norm_layer = (
            nn.GroupNorm(num_groups, base_channels)
            if num_groups != 0
            else nn.BatchNorm2d(base_channels)
        )

        self.inp_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, 1, 3),
            ResNetBlock(base_channels),
            input_norm_layer,
            Swish(beta),
        )

        self.down_blocks = []
        self.up_blocks = []

        for i in range(4):
            self.down_blocks.append(
                DownBlock(
                    base_channels * 2**i,
                    base_channels * 2**i * 2,
                    num_resblocks[i],
                    num_groups,
                    skip_conn_scale,
                    beta,
                    downsample_conv=downsample_conv,
                )
            )
            self.up_blocks.append(
                UpBlock(
                    base_channels * 2 ** (4 - i),
                    base_channels * 2 ** (4 - i) // 2,
                    num_resblocks[::-1][i],
                    num_groups,
                    skip_conn_scale,
                    beta,
                    use_pixelshuffle=use_pixelshuffle,
                )
            )
        self.down_blocks = nn.ModuleList(self.down_blocks)

        self.up_blocks = nn.ModuleList(self.up_blocks)

        if upscale:
            self.upscale = UpBlock(
                base_channels,
                base_channels,
                1,
                num_groups,
                skip_conn_scale,
                beta,
                use_pixelshuffle=use_pixelshuffle,
            )
        else:
            self.upscale = nn.Identity()

        output_norm_layer = (
            nn.GroupNorm(num_groups, base_channels)
            if num_groups != 0
            else nn.BatchNorm2d(base_channels)
        )

        self.out_conv = nn.Sequential(
            output_norm_layer, Swish(beta), nn.Conv2d(base_channels, 3, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Function for Unet

        Args:
            x (torch.Tensor): Input Image

        Returns:
            torch.Tensor: Output Image
        """
        downblock_outputs = []
        x = self.inp_conv(x)
        downblock_outputs.append(x.clone())

        # down sample
        for downblock in self.down_blocks:
            x = downblock(x)
            downblock_outputs.append(x.clone())

        # upsample
        for i, upblock in enumerate(self.up_blocks):
            x = upblock(x) + downblock_outputs[::-1][i + 1]

        # upscale
        x = self.upscale(x)

        out = self.out_conv(x)

        return out
