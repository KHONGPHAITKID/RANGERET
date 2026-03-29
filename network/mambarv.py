import torch
import torch.nn.functional as F
from torch import nn

from mamba_ssm import Mamba

from network.interfaces import SegmentationModel


class CircularConv2d(nn.Module):
    """Convolution with circular padding on width and zero padding on height."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )
        self.pad_h = padding[0]
        self.pad_w = padding[1]

    def forward(self, x):
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0)
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_circular_padding=True):
        super().__init__()
        conv_cls = CircularConv2d if use_circular_padding else nn.Conv2d
        if use_circular_padding:
            self.conv = conv_cls(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv = conv_cls(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Mamba2DBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        bsz, channels, height, width = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm(x_flat)
        out_flat = self.mamba(x_flat)
        out = out_flat.transpose(1, 2).reshape(bsz, channels, height, width)
        return out + x


class MambaDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_circular_padding=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels, use_circular_padding=use_circular_padding)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MambaRV(SegmentationModel):
    """
    CNN encoder-decoder with Mamba bottleneck blocks for range-view segmentation.
    Returns logits in channels-first format `(B, C, H, W)`.
    """

    def __init__(self, model_params, resolution, num_classes=20, activate_recurrent=False):
        super().__init__()
        del activate_recurrent  # kept for factory compatibility

        in_channels = model_params.get("input_dim", 5)
        encoder_channels = model_params.get("encoder_channels", [32, 64, 128, 256])
        decoder_channels = model_params.get("decoder_channels", [128, 64, 32])
        use_circular_padding = model_params.get("use_circular_padding", True)

        if len(encoder_channels) != 4:
            raise ValueError("MambaRV expects exactly 4 encoder channel values.")
        if len(decoder_channels) != 3:
            raise ValueError("MambaRV expects exactly 3 decoder channel values.")

        mamba_cfg = model_params.get("mamba", {})
        mamba_depth = mamba_cfg.get("depth", 3)
        d_state = mamba_cfg.get("d_state", 16)
        d_conv = mamba_cfg.get("d_conv", 4)
        expand = mamba_cfg.get("expand", 2)

        c1, c2, c3, c4 = encoder_channels
        d3, d2, d1 = decoder_channels

        self.input_resolution = resolution
        self.enc1 = ConvBlock(in_channels, c1, stride=1, use_circular_padding=use_circular_padding)
        self.enc2 = ConvBlock(c1, c2, stride=2, use_circular_padding=use_circular_padding)
        self.enc3 = ConvBlock(c2, c3, stride=2, use_circular_padding=use_circular_padding)
        self.enc4 = ConvBlock(c3, c4, stride=2, use_circular_padding=use_circular_padding)

        self.mamba_bottleneck = nn.Sequential(
            *[Mamba2DBlock(c4, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(mamba_depth)]
        )

        self.dec3 = MambaDecoderBlock(c4, c3, d3, use_circular_padding=use_circular_padding)
        self.dec2 = MambaDecoderBlock(d3, c2, d2, use_circular_padding=use_circular_padding)
        self.dec1 = MambaDecoderBlock(d2, c1, d1, use_circular_padding=use_circular_padding)
        self.final_conv = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, x):
        input_spatial = x.shape[-2:]

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        bottleneck = self.mamba_bottleneck(x4)

        d3 = self.dec3(bottleneck, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        logits = self.final_conv(d1)

        if logits.shape[-2:] != input_spatial:
            logits = F.interpolate(logits, size=input_spatial, mode="bilinear", align_corners=False)

        return {"logits": logits, "aux": {}}
