import math

import torch
from torch import nn

from config import get_settings


class InputEmbedding(nn.Module):
    """Source (encoder-side) token embedding, scaled by sqrt(d_model)."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model
        self.embedding = nn.Embedding(cfg.src_vocab_size, cfg.d_model)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Source token ids of shape (batch_size, src_len)

        Returns:
            Embeddings of shape (batch_size, src_len, d_model)
        """
        return self.embedding(tokens) * self.scale
