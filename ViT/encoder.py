import torch
from torch import nn

from config import get_settings
from feedForward import FeedForward
from multiheadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    One pre-norm Vision Transformer encoder layer (Eq. 2–3 of the paper):

        z'ₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁
        zₗ  = MLP(LN(z'ₗ)) + z'ₗ

    LayerNorm comes *before* each block, and the residual adds the block's output back to its
    unnormalized input (the original Transformer normalizes *after* the residual instead).
    """

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        # 1. LayerNorm → multi-head self-attention
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.self_attn = MultiHeadAttention()
        self.dropout1 = nn.Dropout(cfg.dropout)

        # 2. LayerNorm → MLP (Linear → GELU → Linear)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward()
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_patches + 1, d_model)
        """
        attn_output, _ = self.self_attn(self.norm1(x))
        x = x + self.dropout1(attn_output)             # residual around attention

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)              # residual around the MLP
        return x


class Encoder(nn.Module):
    """The stack of L encoder layers, followed by the final LayerNorm (y = LN(z_L))."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        self.layers = nn.ModuleList(
            [EncoderLayer() for _ in range(cfg.n_encoder_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_patches + 1, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
