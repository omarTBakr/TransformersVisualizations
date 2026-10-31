from typing import Optional

import torch
from torch import nn

from config import get_settings
from decoder import Decoder
from encoder import Encoder
from inputEmbedding import InputEmbedding
from outputEmbedding import OutputEmbedding
from positionalEncoding import PositionalEncoding


class Transformer(nn.Module):
    """
    End-to-end Sequence-to-Sequence Transformer Model.
    All architectural hyperparameters are read from config.yml.

    Mask convention everywhere: boolean masks use True for positions that must NOT be
    attended to; float masks are added to the attention scores.
    """

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        # 1. Embeddings & Positional Encodings
        self.input_embedding = InputEmbedding()
        self.output_embedding = OutputEmbedding()
        self.positional_encoding = PositionalEncoding()

        # 2. Encoder & Decoder Stacks
        self.encoder = Encoder()
        self.decoder = Decoder()

        # 3. Final linear projection to vocabulary
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)

        self._reset_parameters()

        # Optional: share weights between output embedding and vocabulary projection.
        # Done after initialization so both modules point to the same tensor.
        if cfg.share_embeddings:
            self.generator.weight = self.output_embedding.weight

    def _reset_parameters(self) -> None:
        """Xavier-uniform init for all weight matrices, as in the original Transformer."""
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(
        size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Causal (size, size) mask where True denotes masked (future) positions."""
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    @staticmethod
    def make_padding_mask(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """(batch_size, seq_len) mask where True denotes padding tokens."""
        return tokens == pad_token_id

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes source token sequence.

        Args:
            src: (batch_size, src_len)

        Returns:
            memory: (batch_size, src_len, d_model)
        """
        x = self.positional_encoding(self.input_embedding(src))
        return self.encoder(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decodes target token sequence conditioned on encoder memory.
        A causal mask is created automatically when ``tgt_mask`` is None.

        Args:
            tgt: (batch_size, tgt_len)
            memory: (batch_size, src_len, d_model)

        Returns:
            Decoder hidden states: (batch_size, tgt_len, d_model)
        """
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        x = self.positional_encoding(self.output_embedding(tgt))
        return self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            enc_padding_mask=enc_padding_mask,
            memory_mask=memory_mask,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass through Transformer.

        Args:
            src: Source tokens of shape (batch_size, src_len)
            tgt: Target tokens of shape (batch_size, tgt_len)
            src_mask: Optional mask for source self-attention (src_len, src_len)
            tgt_mask: Optional mask for target self-attention; causal if None
            src_key_padding_mask: Optional padding mask for source (batch_size, src_len)
            tgt_key_padding_mask: Optional padding mask for target (batch_size, tgt_len)
            memory_mask: Optional mask for cross-attention (tgt_len, src_len)

        Returns:
            Logits of shape (batch_size, tgt_len, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        dec_output = self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            enc_padding_mask=src_key_padding_mask,
            memory_mask=memory_mask,
        )
        return self.generator(dec_output)
