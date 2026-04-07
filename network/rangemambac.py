import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from network.interfaces import SegmentationModel


class RMSNorm2D(nn.Module):
    """RMSNorm over the channel dimension for tensors shaped (B, C, H, W)."""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, act_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer if act_layer is not None else nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(out_channels // 2, 1)
        self.pre = ConvBNAct(
            in_channels,
            mid_channels,
            kernel_size=1,
            padding=0,
            act_layer=nn.LeakyReLU(0.1, inplace=True),
        )
        self.branch1 = ConvBNAct(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_layer=nn.LeakyReLU(0.1, inplace=True),
        )
        self.branch2 = ConvBNAct(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            act_layer=nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = ConvBNAct(
            mid_channels * 2,
            out_channels,
            kernel_size=1,
            padding=0,
            act_layer=nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj = nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.proj(x)
        x = self.pre(x)
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.fuse(x) + identity


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=1, act_layer=nn.LeakyReLU(0.1, inplace=True)),
            ConvBNAct(channels, channels, kernel_size=3, padding=1, act_layer=nn.LeakyReLU(0.1, inplace=True)),
        )

    def forward(self, x):
        return x + self.block(x)


class RangeMambaStem(nn.Module):
    def __init__(self, in_channels=5, out_channels=96):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, padding=1, act_layer=nn.LeakyReLU(0.1, inplace=True)),
            ContextBlock(32, 64),
            ContextBlock(64, out_channels),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x):
        return self.blocks(x)


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = RMSNorm2D(out_channels)

    def forward(self, x):
        return self.norm(self.proj(x))


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            ConvBNAct(out_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return self.block(x)


class MambaSequenceLayer(nn.Module):
    """Thin wrapper around mamba_ssm.Mamba for sequence tensors shaped (N, L, C)."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, force_fp32=False):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32 or x.dtype == torch.float32:
            return self.mamba(x)

        with torch.cuda.amp.autocast(enabled=False):
            return self.mamba(x.float())


class CircularRowBiMamba(nn.Module):
    def __init__(self, dim, pad_len=16, d_state=16, d_conv=4, expand=2, force_fp32=False):
        super().__init__()
        self.pad_len = pad_len
        self.forward_mamba = MambaSequenceLayer(dim, d_state=d_state, d_conv=d_conv, expand=expand, force_fp32=force_fp32)
        self.backward_mamba = MambaSequenceLayer(dim, d_state=d_state, d_conv=d_conv, expand=expand, force_fp32=force_fp32)
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(batch_size * height, width, channels)

        pad_len = min(self.pad_len, width)
        if pad_len > 0:
            seq = torch.cat([seq[:, -pad_len:, :], seq, seq[:, :pad_len, :]], dim=1)

        y_forward = self.forward_mamba(seq)
        y_backward = torch.flip(self.backward_mamba(torch.flip(seq, dims=[1])), dims=[1])

        if pad_len > 0:
            y_forward = y_forward[:, pad_len:pad_len + width, :]
            y_backward = y_backward[:, pad_len:pad_len + width, :]

        y = self.fuse(torch.cat([y_forward, y_backward], dim=-1))
        return y.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


class ColBiMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, force_fp32=False):
        super().__init__()
        self.forward_mamba = MambaSequenceLayer(dim, d_state=d_state, d_conv=d_conv, expand=expand, force_fp32=force_fp32)
        self.backward_mamba = MambaSequenceLayer(dim, d_state=d_state, d_conv=d_conv, expand=expand, force_fp32=force_fp32)
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        y_forward = self.forward_mamba(seq)
        y_backward = torch.flip(self.backward_mamba(torch.flip(seq, dims=[1])), dims=[1])
        y = self.fuse(torch.cat([y_forward, y_backward], dim=-1))

        return y.reshape(batch_size, width, height, channels).permute(0, 3, 2, 1).contiguous()


class ConvFFN(nn.Module):
    def __init__(self, dim, expand=2, dropout=0.0):
        super().__init__()
        hidden_dim = dim * expand
        self.expand = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.expand(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.dropout(x)
        return self.project(x)


class CircularAxialMambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        row_pad_len=16,
        d_state=16,
        d_conv=4,
        mamba_expand=2,
        ffn_expand=2,
        dropout=0.0,
        force_mamba_fp32=False,
        layer_scale_init=1e-3,
    ):
        super().__init__()
        self.norm1 = RMSNorm2D(dim)
        self.local_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.row_mixer = CircularRowBiMamba(
            dim,
            pad_len=row_pad_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
            force_fp32=force_mamba_fp32,
        )
        self.col_mixer = ColBiMamba(dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand, force_fp32=force_mamba_fp32)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.norm2 = RMSNorm2D(dim)
        self.ffn = ConvFFN(dim, expand=ffn_expand, dropout=dropout)
        self.layer_scale_mixer = nn.Parameter(torch.ones(1, dim, 1, 1) * layer_scale_init)
        self.layer_scale_ffn = nn.Parameter(torch.ones(1, dim, 1, 1) * layer_scale_init)

    def forward(self, x):
        h = self.norm1(x)
        local = self.local_dwconv(h)
        row = self.row_mixer(h)
        col = self.col_mixer(h)
        x = x + self.layer_scale_mixer * self.fuse(torch.cat([row, col, local], dim=1))
        return x + self.layer_scale_ffn * self.ffn(self.norm2(x))


class DecoderFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, padding=1),
            ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class FPNDecoder(nn.Module):
    def __init__(self, c3=256, c2=192, c1=128, c0=96):
        super().__init__()
        self.proj3 = nn.Conv2d(c3, c2, kernel_size=1, bias=False)
        self.fuse2 = DecoderFuse(c2 + c2, c2)
        self.proj2 = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.fuse1 = DecoderFuse(c2 + c1, c1)
        self.proj1 = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.fuse0 = DecoderFuse(c1 + c0, c0)

    def forward(self, f3, skip2, skip1, skip0):
        x = self.proj3(f3)
        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, skip2], dim=1))

        x = self.proj2(x)
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([x, skip1], dim=1))

        x = self.proj1(x)
        x = F.interpolate(x, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse0(torch.cat([x, skip0], dim=1))


class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class RangeMambaC(SegmentationModel):
    """
    Hierarchical Circular Axial Mamba model for range-view LiDAR segmentation.

    Default SemanticKITTI flow:
    (B, 5, 64, 1024) -> (B, 96, 64, 1024) -> (B, 128, 16, 256)
    -> (B, 192, 8, 128) -> (B, 256, 4, 64) -> (B, K, 64, 1024).
    """

    def __init__(self, model_params, resolution, num_classes=20, activate_recurrent=False):
        super().__init__()
        del activate_recurrent

        input_dim = model_params.get("input_dim", 5)
        stem_dim = model_params.get("stem_dim", 96)
        stage_dims = model_params.get("stage_dims", [128, 192, 256])
        stage_depths = model_params.get("stage_depths", [3, 4, 6])
        row_pad = model_params.get("row_pad", [16, 8, 4])
        dropout = model_params.get("dropout", 0.0)
        ffn_expand = model_params.get("ffn_expand", 2)
        force_mamba_fp32 = model_params.get("force_mamba_fp32", True)
        layer_scale_init = model_params.get("layer_scale_init", 1e-3)
        use_boundary_head = model_params.get("use_boundary_head", False)

        if len(stage_dims) != 3:
            raise ValueError("RangeMambaC expects exactly 3 stage dimensions.")
        if len(stage_depths) != 3:
            raise ValueError("RangeMambaC expects exactly 3 stage depths.")
        if len(row_pad) != 3:
            raise ValueError("RangeMambaC expects exactly 3 row padding values.")

        mamba_cfg = model_params.get("mamba", {})
        d_state = mamba_cfg.get("d_state", 16)
        d_conv = mamba_cfg.get("d_conv", 4)
        mamba_expand = mamba_cfg.get("expand", 2)

        patch_kernel = model_params.get("patch_kernel", 7)
        patch_stride = model_params.get("patch_stride", 4)
        patch_padding = model_params.get("patch_padding", 3)

        c1, c2, c3 = stage_dims
        d1, d2, d3 = stage_depths
        p1, p2, p3 = row_pad

        self.input_resolution = resolution
        self.stem = RangeMambaStem(input_dim, stem_dim)
        self.patch1 = PatchEmbed2D(stem_dim, c1, kernel_size=patch_kernel, stride=patch_stride, padding=patch_padding)
        self.stage1 = self._make_stage(d1, c1, p1, d_state, d_conv, mamba_expand, ffn_expand, dropout, force_mamba_fp32, layer_scale_init)
        self.down12 = Downsample2D(c1, c2)
        self.stage2 = self._make_stage(d2, c2, p2, d_state, d_conv, mamba_expand, ffn_expand, dropout, force_mamba_fp32, layer_scale_init)
        self.down23 = Downsample2D(c2, c3)
        self.stage3 = self._make_stage(d3, c3, p3, d_state, d_conv, mamba_expand, ffn_expand, dropout, force_mamba_fp32, layer_scale_init)
        self.decoder = FPNDecoder(c3=c3, c2=c2, c1=c1, c0=stem_dim)
        self.seg_head = SegHead(stem_dim, num_classes)
        self.boundary_head = BoundaryHead(stem_dim) if use_boundary_head else None

    @staticmethod
    def _make_stage(depth, dim, row_pad_len, d_state, d_conv, mamba_expand, ffn_expand, dropout, force_mamba_fp32, layer_scale_init):
        return nn.Sequential(
            *[
                CircularAxialMambaBlock(
                    dim,
                    row_pad_len=row_pad_len,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    ffn_expand=ffn_expand,
                    dropout=dropout,
                    force_mamba_fp32=force_mamba_fp32,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        input_spatial = x.shape[-2:]

        skip0 = self.stem(x)
        skip1 = self.stage1(self.patch1(skip0))
        skip2 = self.stage2(self.down12(skip1))
        x = self.stage3(self.down23(skip2))

        features = self.decoder(x, skip2, skip1, skip0)
        logits = self.seg_head(features)
        if logits.shape[-2:] != input_spatial:
            logits = F.interpolate(logits, size=input_spatial, mode="bilinear", align_corners=False)

        aux = {}
        if self.boundary_head is not None:
            boundary = self.boundary_head(features)
            if boundary.shape[-2:] != input_spatial:
                boundary = F.interpolate(boundary, size=input_spatial, mode="bilinear", align_corners=False)
            aux["boundary"] = boundary

        return {"logits": logits, "aux": aux}
