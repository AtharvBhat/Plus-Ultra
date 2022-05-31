import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self, swish=1.0) -> None:
        super(Swish, self).__init__()
        self.swish = swish

    def forward(self, x):
        if self.swish == 1.0:
            return F.silu(x)
        else:
            return x * F.sigmoid(x * float(self.swish))

class ResNetBlock(nn.Module):
    def __init__(self, channels, num_groups = 32, skip_connection_scale=1, swish=1.0, skip_path=False):
        super(ResNetBlock, self).__init__()
        
        if num_groups != 0:
            self.main_path = nn.Sequential(
                nn.GroupNorm(num_groups, channels),
                Swish(swish),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.GroupNorm(num_groups, channels),
                Swish(swish),
                nn.Conv2d(channels, channels, 3, 1, 1)
            )
        else:
            self.main_path = nn.Sequential(
                nn.BatchNorm2d(channels),
                Swish(swish),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.BatchNorm2d(channels),
                Swish(swish),
                nn.Conv2d(channels, channels, 3, 1, 1)
            )
        self.skip_path = skip_path
        if skip_path:
            self.skip = nn.Conv2d(channels, channels, 1, 1, 0)
        self.skip_conn_scale = skip_connection_scale
    
    def forward(self, x):
        x_main = self.main_path(x)
        if self.skip_path:
            x_skip = self.skip(x)
        else:
            x_skip = x
        return x_main + x_skip * self.skip_conn_scale

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, numResNetBlocks, num_groups = 32, skip_connection_scale=1, swish=1.0, use_conv=False, skip_path=False) -> None:
        super(DBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.use_conv = use_conv
        if use_conv:
            self.down_sample_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        else:
            self.down_sample_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.resblocks = nn.ModuleList([ResNetBlock(out_channels, num_groups, skip_connection_scale, swish, skip_path=skip_path) for i in range(numResNetBlocks)])

    def forward(self, x):
        if not self.use_conv:
            x = self.max_pool(x)
        x = self.down_sample_conv(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, numResNetBlocks, num_groups = 32, skip_connection_scale=1, swish=1.0, skip_path=False) -> None:
        super(UBlock, self).__init__()
        self.upblock = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.resblocks = nn.ModuleList([ResNetBlock(in_channels, num_groups, skip_connection_scale, swish, skip_path=skip_path) for i in range(numResNetBlocks)])

    def forward(self, x):
        for resblock in self.resblocks:
            x = resblock(x) 
        x = self.upblock(x)    

        return x 

class Unet(nn.Module):
    def __init__(self, config) -> None:
        super(Unet, self).__init__()
        #init model with config
        base_channels = config["base_channels"]
        res1, res2, res3, res4 = config["resblocks"]
        swish = config["swish"]
        num_groups = config["num_groups"]
        skip_connection_scale = config["skip_connection_scale"]
        downsample_conv = config["down_conv"]
        skip_conv = config["skip_conv"]

        #downsample blocks
        if num_groups!=0:
            self.inp_conv = nn.Sequential(nn.Conv2d(3, base_channels, 3, 1, 1),
                                        ResNetBlock(base_channels),
                                        nn.GroupNorm(num_groups, base_channels),
                                        Swish(swish))
        else:
            self.inp_conv = nn.Sequential(nn.Conv2d(3, base_channels, 3, 1, 1),
                                        ResNetBlock(base_channels),
                                        nn.BatchNorm2d(base_channels),
                                        Swish(swish))
        self.down_1 = DBlock(base_channels, base_channels*2, res1, num_groups, skip_connection_scale, swish, use_conv=downsample_conv, skip_path=skip_conv) 
        self.down_2 = DBlock(base_channels*2, base_channels*4, res2, num_groups, skip_connection_scale, swish, use_conv=downsample_conv, skip_path=skip_conv)
        self.down_3 = DBlock(base_channels*4, base_channels*8, res3, num_groups, skip_connection_scale, swish, use_conv=downsample_conv, skip_path=skip_conv)
        self.down_4 = DBlock(base_channels*8, base_channels*16, res4, num_groups, skip_connection_scale, swish, use_conv=downsample_conv, skip_path=skip_conv)

        #upsample blocks
        self.up_1 = UBlock(base_channels*16, base_channels*8, res4, num_groups, skip_connection_scale, swish, skip_path=skip_conv)
        self.up_2 = UBlock(base_channels*8, base_channels*4, res3, num_groups, skip_connection_scale, swish, skip_path=skip_conv)
        self.up_3 = UBlock(base_channels*4, base_channels*2, res2, num_groups, skip_connection_scale, swish, skip_path=skip_conv)
        self.up_4 = UBlock(base_channels*2, base_channels, res1, num_groups, skip_connection_scale, swish, skip_path=skip_conv)
        self.up_5 = UBlock(base_channels, base_channels//2, 1, num_groups, skip_connection_scale, swish, skip_path=skip_conv)
        
        if num_groups!=0:
            self.out_conv = nn.Sequential(nn.GroupNorm(num_groups, base_channels//2),
                                        Swish(swish),
                                        nn.Conv2d(base_channels//2, 3, 3, 1, 1))
        else:
            self.out_conv = nn.Sequential(nn.BatchNorm2d(base_channels//2),
                                        Swish(swish),
                                        nn.Conv2d(base_channels//2, 3, 3, 1, 1))
    
    def forward(self, x):
        # assuming 3 x 256 x 256 input
        x = self.inp_conv(x) # output: 64 x 256 x 256
        #down sample
        x_d1 = self.down_1(x) # output : 128 x 128 x 128
        x_d2 = self.down_2(x_d1) # output : 256 x 64 x 64
        x_d3 = self.down_3(x_d2) # output : 512 x 32 x 32
        x_d4 = self.down_4(x_d3) # output : 1024 x 16 x 16
        
        #upsample
        x_up1 = self.up_1(x_d4) # output : 512 x 32 x 32
        x_up2 = self.up_2(x_up1 + x_d3) #output : 256 x 64 x 64
        x_up3 = self.up_3(x_up2 + x_d2) # output : 128 x 128 x 128
        x_up4 = self.up_4(x_up3 + x_d1) # output : 64 x 256 x 256
        x_up5 = self.up_5(x_up4) # output : 32 x 512 x 512

        out = self.out_conv(x_up5)

        return out