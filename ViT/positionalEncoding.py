import math

import torch
from torch import nn

from config import get_settings


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings, followed by dropout.

    Supported ``positional_encoding`` values in config.yml:
      - ``"sinusoidal"``: fixed encodings from 'Attention Is All You Need'
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
      - ``"learned"``: a trainable embedding per position.
    """

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        self.encoding_type = cfg.positional_encoding
        self.dropout = nn.Dropout(p=cfg.dropout)

        if self.encoding_type == "sinusoidal":
            # persistent=False: deterministic, so no need to store it in checkpoints.
            self.register_buffer("pe", self._build_sinusoidal(), persistent=False)
        elif self.encoding_type == "learned":
            self.pe = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
            nn.init.normal_(self.pe, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unsupported positional encoding: {self.encoding_type}")

    def _build_sinusoidal(self) -> torch.Tensor:
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(self.max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Compute 1/10000^(2i/d) in log-space to avoid float overflow:
        # exp(-2i/d * log(10000)) instead of 10000^(-2i/d)
        div_term = torch.exp(
            -torch.arange(0, self.d_model, 2).float() / self.d_model * torch.log(torch.tensor(10000.0))
        )  # shape: (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        # odd dims; with an odd d_model there is one fewer cosine column than sine column
        pe[:, 1::2] = torch.cos(position * div_term[: self.d_model // 2])
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encodings added.
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        x = x + self.pe[:, :seq_len].to(dtype=x.dtype)
        return self.dropout(x)
