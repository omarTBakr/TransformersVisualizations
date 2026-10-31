from typing import Optional

import torch
from torch import nn

from config import get_settings
from feedForward import FeedForward
from multiheadAttention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """A single (post-norm) Transformer Decoder Layer configured via config.yml."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        # 1. Masked Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout1 = nn.Dropout(cfg.dropout)

        # 2. Multi-Head Cross-Attention (attending over encoder memory)
        self.cross_attn = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout2 = nn.Dropout(cfg.dropout)

        # 3. Position-wise Feed-Forward Network
        self.ffn = FeedForward()
        self.norm3 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Target representations (batch_size, tgt_len, d_model)
            enc_output: Encoder outputs (batch_size, src_len, d_model)
            tgt_mask: Causal mask for target self-attention (tgt_len, tgt_len), True = masked
            tgt_key_padding_mask: Padding mask for target sequence (batch_size, tgt_len)
            enc_padding_mask: Padding mask for encoder key sequence (batch_size, src_len)
            memory_mask: Optional cross-attention memory mask (tgt_len, src_len)
        """
        # 1. Masked self-attention
        attn1_output, _ = self.self_attn(
            x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout1(attn1_output))

        # 2. Cross-attention over encoder outputs
        attn2_output, _ = self.cross_attn(
            x,
            enc_output,
            enc_output,
            attn_mask=memory_mask,
            key_padding_mask=enc_padding_mask,
        )
        x = self.norm2(x + self.dropout2(attn2_output))

        # 3. Feed-forward network
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class Decoder(nn.Module):
    """Full Transformer Decoder stack consisting of n_decoder_layers."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        self.layers = nn.ModuleList(
            [DecoderLayer() for _ in range(cfg.n_decoder_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Passes representations through all decoder layers."""
        for layer in self.layers:
            x = layer(
                x,
                enc_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                enc_padding_mask=enc_padding_mask,
                memory_mask=memory_mask,
            )
        return self.norm(x)
