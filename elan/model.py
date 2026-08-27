"""
ELAN — Efficient Long-Range Attention Network for image super-resolution.

A compact, faithful reimplementation of the architecture from
"Efficient Long-Range Attention Network for Image Super-resolution"
(Zhang et al., ECCV 2022), sized for trading-card upscaling.

Building blocks:
- GMSA: group multi-scale self-attention. Channels are split into groups;
  attention maps computed on the first group are *shared* with the others,
  which is where most of the compute saving comes from.
- ShiftConvFFN: 1x1 expand -> shift-conv (cheap long-range mixing) -> 1x1.
- ELAB: LayerNorm -> GMSA -> residual, LayerNorm -> ShiftConvFFN -> residual,
  each with a learnable per-channel layer scale.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*num_windows, window_size*window_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(B*num_windows, window_size*window_size, C) -> (B, H, W, C)."""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


def _pad_to_multiple(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad H/W (channel-last input) up to a multiple of window_size."""
    _, H, W, _ = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, (H, W)


# --------------------------------------------------------------------------- #
# Window attention with relative position bias (Swin-style)
# --------------------------------------------------------------------------- #

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Relative position bias table: (2*ws-1)^2 entries per axis.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (attended_values, attention_probs). x: (nW*B, N, C)."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return out, attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._attention(x)
        return out


class GMSA(nn.Module):
    """
    Group multi-scale self-attention with shared attention maps.

    Channels are split into `n_groups` groups. Attention probabilities are
    computed once (on the first group) and reused for the remaining groups,
    matching ELAN's shared-attention trick. Consecutive ELABs can be given
    different window sizes to mix local and long-range context.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        n_groups: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()
        if dim % n_groups != 0:
            raise ValueError(f"dim {dim} must divide evenly into {n_groups} groups")
        self.n_groups = n_groups
        self.window_size = window_size
        self.attn = WindowAttention(dim // n_groups, window_size, num_heads, qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) channel-last."""
        B, H, W, C = x.shape
        x, (H0, W0) = _pad_to_multiple(x, self.window_size)
        Hp, Wp = x.shape[1], x.shape[2]

        groups = x.chunk(self.n_groups, dim=-1)
        windows = [window_partition(g, self.window_size) for g in groups]

        # Shared attention: compute probs on group 0, reuse everywhere.
        out0, attn = self.attn._attention(windows[0])
        outs = [out0]
        for w in windows[1:]:
            B_, N, Cg = w.shape
            v = self.attn.qkv(w).reshape(B_, N, 3, self.attn.num_heads, Cg // self.attn.num_heads)
            v = v.permute(2, 0, 3, 1, 4)[2]
            outs.append((attn @ v).transpose(1, 2).reshape(B_, N, Cg))

        merged = torch.cat(
            [window_reverse(o, self.window_size, Hp, Wp) for o in outs], dim=-1
        )
        merged = self.proj(merged)
        if (Hp, Wp) != (H0, W0):
            merged = merged[:, :H0, :W0, :]
        return merged.contiguous()


# --------------------------------------------------------------------------- #
# Shift-conv FFN — cheap spatial mixing without extra attention
# --------------------------------------------------------------------------- #

class ShiftConv(nn.Module):
    """3x3 conv preceded by 5-way channel shifting (center + 4 directions)."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=1)
        # 5 shifts: identity, up, down, left, right — channels split as evenly as possible
        base = dim // 5
        sizes = [base] * 5
        for i in range(dim - base * 5):
            sizes[i] += 1
        self.sizes = sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        chunks = torch.split(x, self.sizes, dim=1)
        shifted = [
            chunks[0],
            torch.roll(chunks[1], shifts=1, dims=2),
            torch.roll(chunks[2], shifts=-1, dims=2),
            torch.roll(chunks[3], shifts=1, dims=3),
            torch.roll(chunks[4], shifts=-1, dims=3),
        ]
        return self.conv(torch.cat(shifted, dim=1))


class ShiftConvFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.shift_conv = ShiftConv(hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        return self.fc2(self.shift_conv(self.act(self.fc1(x))))


# --------------------------------------------------------------------------- #
# ELAB block
# --------------------------------------------------------------------------- #

class ELAB(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        n_groups: int = 4,
        ffn_expansion: float = 2.0,
        layer_scale: float = 1e-2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GMSA(dim, window_size, num_heads, n_groups)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ShiftConvFFN(dim, ffn_expansion)
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        shortcut = x
        y = self.norm1(x.permute(0, 2, 3, 1))
        y = self.attn(y)
        y = y.permute(0, 3, 1, 2)
        x = shortcut + self.gamma1.view(1, -1, 1, 1) * y

        y = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.gamma2.view(1, -1, 1, 1) * self.ffn(y)
        return x


# --------------------------------------------------------------------------- #
# Full network
# --------------------------------------------------------------------------- #

class ELAN(nn.Module):
    """
    ELAN super-resolution network.

    Default sizes follow the paper's full model (channels=60, n_blocks=24);
    pass smaller values for quick experiments or CPU training.
    """

    def __init__(
        self,
        upscale: int = 2,
        in_chans: int = 3,
        channels: int = 60,
        n_blocks: int = 24,
        window_sizes: Optional[Sequence[int]] = None,
        num_heads: int = 5,
        n_groups: int = 4,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()
        if channels % n_groups != 0:
            raise ValueError("channels must divide evenly into n_groups")
        self.upscale = upscale
        self.window_sizes: List[int] = list(window_sizes) if window_sizes else [8, 8, 8, 16]

        self.conv_in = nn.Conv2d(in_chans, channels, 3, padding=1)
        self.body = nn.ModuleList(
            ELAB(
                dim=channels,
                window_size=self.window_sizes[i % len(self.window_sizes)],
                num_heads=num_heads,
                n_groups=n_groups,
                ffn_expansion=ffn_expansion,
            )
            for i in range(n_blocks)
        )
        self.conv_body = nn.Conv2d(channels, channels, 3, padding=1)

        # Upsampler: one conv+PixelShuffle(2) per factor of two.
        n_up = int(round(math.log2(upscale)))
        if 2 ** n_up != upscale:
            raise ValueError(f"upscale must be a power of two, got {upscale}")
        up_layers = []
        for _ in range(n_up):
            up_layers += [nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2)]
        up_layers += [nn.Conv2d(channels, in_chans, 3, padding=1)]
        self.upsampler = nn.Sequential(*up_layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1] -> (B, 3, H*scale, W*scale)."""
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        feat = self.conv_in(x)
        body = feat
        for block in self.body:
            body = block(body)
        body = self.conv_body(body) + feat
        return self.upsampler(body) + base


class ELANLight(ELAN):
    """Lightweight variant (paper: channels=36, n_blocks=12)."""

    def __init__(self, upscale: int = 2, channels: int = 36, n_blocks: int = 12, **kwargs):
        kwargs.setdefault("num_heads", 3)
        kwargs.setdefault("n_groups", 4)
        super().__init__(
            upscale=upscale, channels=channels, n_blocks=n_blocks, **kwargs
        )
