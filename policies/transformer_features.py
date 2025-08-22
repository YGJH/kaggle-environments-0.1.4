import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerExtractor(BaseFeaturesExtractor):
    """Large Transformer-based feature extractor for ConnectX.

    Observation: flat vector length 42 with values {0,1,2}.
    Encoding:
      - Token embedding for piece type (3 embeddings)
      - Positional embedding (42 embeddings)
      - Optional CLS token prepended
    Architecture:
      - Configurable (default: 32) Transformer encoder layers
      - d_model (hidden size) default 2000
      - n_heads chosen so that d_model % n_heads == 0 (default 10 heads)
      - Mean pooling over sequence -> features_dim

    NOTE: Extremely large; ensure resources are sufficient.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        d_model: int = 2000,
        n_layers: int = 32,
        n_heads: int = 10,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        use_cls: bool = True,
    ):
        super().__init__(observation_space, features_dim=d_model)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.seq_len = 42
        self.use_cls = use_cls

        self.piece_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(self.seq_len, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls else None
        self.dropout = nn.Dropout(dropout)

        ff_dim = ff_multiplier * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.piece_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.long()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.size(0)
        x = x.clamp_(0, 2)
        pos_ids = torch.arange(self.seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        tok = self.piece_embed(x) + self.pos_embed(pos_ids)
        if self.use_cls:
            cls = self.cls_token.expand(batch, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        tok = self.dropout(tok)
        enc = self.encoder(tok)
        feat = enc.mean(dim=1)
        return feat
