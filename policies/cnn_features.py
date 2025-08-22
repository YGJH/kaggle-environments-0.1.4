import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ConnectXCNNExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for 7x6 ConnectX board.

    Input observation expected as flat vector of length 42 with values {0,1,2}.
    We internally build a 2-channel tensor:
        channel 0: our pieces (==1)
        channel 1: opponent pieces (==2)
    Shape fed to conv stack: (B, 2, 6, 7)
    """
    def __init__(self, observation_space: spaces.Space, features_dim: int = 512, hidden_channels=(64,128,128)):
        super().__init__(observation_space, features_dim)
        assert observation_space.shape == (42,), "Expect flat board of length 42"
        c1, c2, c3 = hidden_channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # compute linear input dim (6x7 board)
        with torch.no_grad():
            dummy = torch.zeros(1,2,6,7)
            out = self.conv(dummy)
            flat_dim = out.view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape (B,42) or (42,)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        b = obs.size(0)
        # clamp & build channels
        x = obs.long().clamp_(0,2)
        ours = (x == 1).float().view(b,1,6,7)
        opp = (x == 2).float().view(b,1,6,7)
        board = torch.cat([ours, opp], dim=1)
        feats = self.conv(board)
        feats = self.head(feats)
        return feats
