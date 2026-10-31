import math
from typing import Optional

import torch
from torch import nn

from config import get_settings


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention parameterized by config.yml."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model
        self.dim = cfg.d_model

        # Learnable linear projections for Query, Key, and Value
        self.query_proj = nn.Linear(self.dim, self.dim, bias=cfg.bias)
        self.key_proj = nn.Linear(self.dim, self.dim, bias=cfg.bias)
        self.value_proj = nn.Linear(self.dim, self.dim, bias=cfg.bias)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for single-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            mask: Optional boolean mask broadcastable to (batch_size, seq_len, seq_len).
                  True marks positions that must NOT be attended to (same convention
                  as MultiHeadAttention and torch.nn.MultiheadAttention).

        Returns:
            Attention output tensor of shape (batch_size, seq_len, dim)
        """
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Scaled dot-product: Q @ K^T / sqrt(dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("mask must be a boolean tensor (True = masked)")
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        # Rows where every key is masked yield NaN; zero them out instead.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, value)
