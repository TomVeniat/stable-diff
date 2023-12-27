import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()

        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        return self.linear_2(x)


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, dim: int = 320):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                SwitchSequential(nn.Conv2d(4, dim, kernel_size=3, padding=1)),  # (b, 320, H/8, W/8)
                SwitchSequential(UNetResidualBlock(dim, dim), UNetAttentionBlock(8, 40)),
                SwitchSequential(UNetResidualBlock(dim, dim), UNetAttentionBlock(8, 40)),

                SwitchSequential(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)),  # (b, 320, H/16, W/16)
                SwitchSequential(UNetResidualBlock(dim, 2 * dim), UNetAttentionBlock(8, 80)),  # (b, 640, H/16, W/16)
                SwitchSequential(UNetResidualBlock(2 * dim, 2 * dim), UNetAttentionBlock(8, 80)),

                # (b,640, H/32, W/32)
                SwitchSequential(nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=2, padding=1)),

                # (b,1280, H/32, W/32)
                SwitchSequential(UNetResidualBlock(2 * dim, 4 * dim), UNetAttentionBlock(8, 160)),
                SwitchSequential(UNetResidualBlock(4 * dim, 4 * dim), UNetAttentionBlock(8, 160)),

                # (b, 1280, H/64, W/64)
                SwitchSequential(nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, stride=2, padding=1)),
                SwitchSequential(UNetResidualBlock(4 * dim, 4 * dim)),
                SwitchSequential(UNetResidualBlock(4 * dim, 4 * dim)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(4 * dim, 4 * dim),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(4 * dim, 4 * dim)
        )

        self.decoder = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(8 * dim, 4 * dim)),  # (b, 1280, H/64, W/64)
            SwitchSequential(UNetResidualBlock(8 * dim, 4 * dim)),

            SwitchSequential(UNetResidualBlock(8 * dim, 4 * dim), UpSample(4 * dim)),
            SwitchSequential(UNetResidualBlock(8 * dim, 4 * dim), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(8 * dim, 4 * dim), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(6 * dim, 4 * dim), UNetAttentionBlock(8, 160), UpSample(4 * dim)),

            SwitchSequential(UNetResidualBlock(6 * dim, 2 * dim), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(4 * dim, 2 * dim), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(3 * dim, 3 * dim), UNetAttentionBlock(8, 80), UpSample(2 * dim)),

            SwitchSequential(UNetResidualBlock(3 * dim, dim), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(2 * dim, dim), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(2 * dim, dim), UNetAttentionBlock(8, 40)),
        ])

    def forward(self, x):
        pass


class UNetAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_heads * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels * 2, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)

        # norm + self attention
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # norm + cross attention
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # norm + FF layer and GeGlu
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))
        return self.conv_output(x) + residue_long


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        super().__init__()
        self.groupnorm_features = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(d_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.res_con = nn.Identity()
        else:
            self.res_con = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, features: torch.Tensor, time: torch.Tensor):

        residue = features

        features = self.groupnorm_features(features)
        features = F.silu(features)
        features = self.conv_features(features)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = features + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.res_con(residue)


class UNetOutLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        return self.conv(x)  # (b, 4, H/8, W/8)


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class Diffusion(nn.Module):
    def __init__(self, dim: int = 320):
        super().__init__()
        self.time_embedding = TimeEmbedding(dim)
        self.unet = UNet()
        self.final = UNetOutLayer(dim, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """

        :param latent: (b, 4, H/8, W/8)
        :param context: (b, seq_len, dim)
        :param time: (1, 320)
        :return:
        """

        time = self.time_embedding(time)  # (1,1280)

        out = self.unet(latent, context, time)  # (b, 320, H/8, W/8)

        out = self.final(out)  # (b, 4, H/8, W/8)

        return out
