import math
from typing import Optional, Tuple

import torch
from torch import nn

from config import get_settings


def merge_masks(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    batch_size: int,
    n_heads: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Combine an attention mask and a key padding mask into a single additive float mask
    broadcastable to (batch_size, n_heads, tgt_len, src_len).

    Mask conventions (same as ``torch.nn.MultiheadAttention``):
      - Boolean masks: ``True`` marks a position that must NOT be attended to.
      - Float masks: values are added to the attention scores (use ``-inf`` to block).

    Accepted shapes:
      - attn_mask: (tgt_len, src_len), (batch_size * n_heads, tgt_len, src_len)
        or (batch_size, n_heads, tgt_len, src_len)
      - key_padding_mask: (batch_size, src_len)
    """
    merged: Optional[torch.Tensor] = None

    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.view(batch_size, n_heads, *attn_mask.shape[-2:])
        elif attn_mask.dim() not in (2, 4):
            raise ValueError(f"attn_mask must be 2D, 3D or 4D, got {attn_mask.dim()}D")
        merged = _to_additive(attn_mask, dtype)

    if key_padding_mask is not None:
        if key_padding_mask.dim() != 2:
            raise ValueError("key_padding_mask must have shape (batch_size, src_len)")
        kpm = _to_additive(key_padding_mask, dtype)[:, None, None, :]
        merged = kpm if merged is None else merged + kpm

    return merged


def _to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(
            mask, float("-inf")
        )
    return mask.to(dtype)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module configured using config.yml settings."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        # d_model % n_heads == 0 is guaranteed by ModelConfig validation.
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections for Query, Key, Value, and Output
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=cfg.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=cfg.bias)

        self.dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, n_heads, seq, head_dim)"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention. Supports both self-attention and cross-attention.

        Args:
            query: (batch_size, tgt_len, d_model)
            key: (batch_size, src_len, d_model), defaults to query if None (self-attention)
            value: (batch_size, src_len, d_model), defaults to key if None
            attn_mask: Optional mask, see ``merge_masks`` for accepted shapes/conventions.
            key_padding_mask: Optional (batch_size, src_len) mask; True marks padding.

        Returns:
            Tuple of (output, attention_weights)
            - output: (batch_size, tgt_len, d_model)
            - attention_weights: (batch_size, n_heads, tgt_len, src_len)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, tgt_len, _ = query.shape
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("key and value must share batch size and sequence length")

        # 1. Linear projections, reshaped to (batch_size, n_heads, seq_len, head_dim)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # 2. Scaled dot-product scores: (batch_size, n_heads, tgt_len, src_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Apply attention / padding masks
        mask = merge_masks(attn_mask, key_padding_mask, batch_size, self.n_heads, scores.dtype)
        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        # Rows where every key is masked yield NaN; zero them out instead.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted sum of values, reshaped back to (batch_size, tgt_len, d_model)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)

        return self.out_proj(context), attn_weights
