import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not norm),
        ]
        if norm:
            # Use GroupNorm with 1 group (channel-wise normalization across spatial dims)
            # GroupNorm is spatial-size agnostic and works for inputs (N, C, H, W)
            layers.insert(1, nn.GroupNorm(1, channels))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        out = self.seq(x)
        return torch.relu(out + x)


class ResNetExtractor(BaseFeaturesExtractor):
    """
    ResNet-style features extractor for ConnectX.
    Expects flat observation of length 42 (values 0/1/2). Internally converts
    to two channels: player1 and player2 occupancy, shape (batch,2,6,7).

    Args:
        observation_space: gymnasium space (Box) with shape (42,) or (n,)
        channels: number of conv channels in stem and blocks
        num_blocks: number of residual blocks (each block has 2 conv layers)
        feat_dim: output flattened feature dimension
    """

    def __init__(self, observation_space: spaces.Space, channels: int = 64, num_blocks: int = 12, feat_dim: int = 512):
        # observation_space is expected to be Box(shape=(42,))
        super().__init__(observation_space, features_dim=feat_dim)

        # stem conv producing `channels` feature maps
        self.channels = channels
        self.num_blocks = num_blocks
        self.feat_dim = feat_dim

        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        self.blocks = nn.Sequential(*blocks)

        # Head: global pooling + linear to feat_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, feat_dim),
            nn.ReLU(inplace=True)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, 42) or (42,)
        x = observations.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Convert to two channels: player1 / player2
        # x values are {0,1,2}
        p1 = (x == 1).float()
        p2 = (x == 2).float()
        stacked = torch.stack([p1, p2], dim=1)  # (batch, 2, 42)
        # reshape to (batch,2,6,7)
        batch = stacked.shape[0]
        stacked = stacked.view(batch, 2, 6, 7)

        out = self.stem(stacked)
        out = self.blocks(out)
        out = self.pool(out)
        out = self.fc(out)
        return out
