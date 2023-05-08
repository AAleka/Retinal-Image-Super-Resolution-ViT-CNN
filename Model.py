import math

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
# from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_ratio * dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class To_Image(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

    def forward(self, x):
        x = self.conv(x)

        return x


class ConvolutionBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_activation=True, use_norm=True, **kwargs):
        super(ConvolutionBlockG, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        x = self.convolution(x)
        return x


class ConvolutionBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvolutionBlockD, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class Generator(nn.Module):
    def __init__(self, patch_size, img_channels=3, dim=2048, depth=1, heads=4, dim_head=64, mlp_ratio=4, drop_rate=0.):
        super(Generator, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        dims = (dim, dim // 2, dim // 4, dim // 8, dim // 16, dim // 32)

        self.patch_embed = nn.Conv2d(img_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.TransformerEncoder = Transformer(dim, depth, heads, dim_head, mlp_ratio, drop_rate)

        self.up_blocks = nn.ModuleList()

        i = 0
        for i in range(5):
            self.up_blocks.append(
                ConvolutionBlockG(dims[i], dims[i+1], down=False, kernel_size=3, stride=2, padding=1,
                                  output_padding=1))

        self.last = nn.Conv2d(dims[i+1], img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')

        num_patches1 = x.shape[2] // self.patch_size
        num_patches2 = x.shape[3] // self.patch_size

        x = self.patch_embed(x)
        skip = x
        x = x.flatten(2).transpose(1, 2)

        x = self.TransformerEncoder(x).permute(0, 2, 1).view(-1, self.dim, num_patches1, num_patches2)
        x = skip + x

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)

        x = self.last(x)
        x = torch.tanh(x)

        return x
