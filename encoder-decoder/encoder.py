from typing import Optional

import torch
from torch import nn

from config import get_settings
from feedForward import FeedForward
from multiheadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """A single (post-norm) Transformer Encoder Layer configured via config.yml."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        # 1. Multi-head self-attention
        self.self_attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout1 = nn.Dropout(cfg.dropout)

        # 2. Position-wise Feed-Forward Network
        self.ffn = FeedForward()
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, src_len, d_model)
            src_mask: (src_len, src_len), True = masked
            src_key_padding_mask: (batch_size, src_len), True = padding
        """
        attn_output, _ = self.self_attn(
            x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout1(attn_output))

        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """Full Transformer Encoder stack consisting of n_encoder_layers."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        self.layers = nn.ModuleList(
            [EncoderLayer() for _ in range(cfg.n_encoder_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Passes embedded input through all encoder layers.

        Args:
            x: (batch_size, src_len, d_model)
            src_mask: (src_len, src_len), True = masked
            src_key_padding_mask: (batch_size, src_len), True = padding
        """
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
