import math

import torch
from torch import nn

from config import get_settings


class OutputEmbedding(nn.Module):
    """
    Target (decoder-side) token embedding, scaled by sqrt(d_model).

    Its weight can be tied with the final vocabulary projection
    (see ``share_embeddings`` in config.yml).
    """

    def __init__(self):
        super().__init__()
        cfg = get_settings().model
        self.embedding = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model)
        self.scale = math.sqrt(cfg.d_model)

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Target token ids of shape (batch_size, tgt_len)

        Returns:
            Embeddings of shape (batch_size, tgt_len, d_model)
        """
        return self.embedding(tokens) * self.scale
