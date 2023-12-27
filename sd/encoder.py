import torch
import torch.nn.functional as F
from torch import nn, Tensor

from decoder import VAE_ResidualBlock, VAE_AttentionBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self, dim: int = 128) -> None:
        super().__init__(
            # (b,C, H, W) -> (b, 128, H, W)
            nn.Conv2d(3, dim, kernel_size=3, padding=1),

            # (b, 128, H, W) -> (b, 128, H, W)
            VAE_ResidualBlock(dim, dim),
            VAE_ResidualBlock(dim, dim),

            # (b, 128, H, W) -> (b, 128, H/2, W/2)
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(dim, 2 * dim),
            VAE_ResidualBlock(2 * dim, 2 * dim),

            # (b, 256, H/2, W/2) -> (b, 256, H/4, W/4)
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(2 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),

            # (b, 256, H/4, W/4) -> (b, 256, H/8, W/8)
            nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),
            VAE_ResidualBlock(4 * dim, 4 * dim),

            VAE_AttentionBlock(4 * dim),

            VAE_ResidualBlock(4 * dim, 4 * dim),

            nn.GroupNorm(32, 4 * dim),

            nn.SiLU(),
            nn.Conv2d(2 * dim, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:
        """

        :param x: (b, C, H, W)
        :param noise: (b, out C, H/8, W/8
        :return: z
        """
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (b, 8, H/8, W/8) -> ((b,4,H/8, W/8), (b, 4, H/8, W/8))
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        x = mean + stdev * noise

        x *= 0.18215

        return x
