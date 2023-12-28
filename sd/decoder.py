from torch import nn, Tensor
from torch.nn import functional as F

from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: (b, feats, H, W)
        :return:
        """
        residue = x
        n, c, h, w = x.shape

        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)  # (b, h*w, feats)

        x = self.attention(x)  # (b, h*w, feats)

        x = x.transpose(-1, -2)  # (b, feats, h*w)
        x = x.view(n, c, h, w)

        return x + residue


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self, dim: int = 128):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 4 * dim, kernel_size=3, padding=1),

            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_AttentionBlock(4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, padding=1),
            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, padding=1),
            VAE_ResidualBlock(4 * dim, 2 * dim),
            VAE_ResidualBlock(2 * dim, 2 * dim),
            VAE_ResidualBlock(2 * dim, 2 * dim),

            nn.Upsample(scale_factor=2),  # (b,2*dim, H, W)
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, padding=1),
            VAE_ResidualBlock(2 * dim, 1 * dim),
            VAE_ResidualBlock(dim, dim),
            VAE_ResidualBlock(dim, dim),

            nn.GroupNorm(32, dim),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: (b, 4, H/8, W/8
        :return: (b, 3, H, W)
        """

        x /= 0.18215

        for module in self:
            x = module(x)

        return x  # (b, 3, H, W)
