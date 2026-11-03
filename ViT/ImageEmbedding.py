import torch
from torch import nn

from config import get_settings
from positionalEncoding import PositionalEncoding


class ImageEmbedding(nn.Module):
    """
    Image → sequence of patch embeddings, with a [CLS] token and positional encoding (Eq. 1).

        z₀ = [x_class; x¹_p E; x²_p E; …; xᴺ_p E] + E_pos

    The Conv2d with kernel = stride = patch_size is exactly the paper's linear projection E of
    the flattened patches, computed for every patch at once. The extra Linear after it is
    not in the paper (two linear maps compose into one), and is kept on purpose.
    """

    def __init__(self):
        super().__init__()
        cfg = get_settings().model
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size

        # Patch embedding E: one d_model-dim vector per (P × P × C) patch
        self.patch_embedding = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        # Extra projection of the patch embeddings (not in the paper)
        self.linear_embedding = nn.Linear(cfg.d_model, cfg.d_model)
        # The learnable [class] token x_class, prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        # Positional encoding (sinusoidal or learned, from config.yml); adds to x and applies dropout
        self.positional_encoding = PositionalEncoding()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images of shape (batch_size, channels, height, width)

        Returns:
            Token embeddings of shape (batch_size, n_patches + 1, d_model); token 0 is [CLS].
        """
        batch_size, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected {self.image_size}×{self.image_size} images, got {height}×{width}")

        # (B, C, H, W) -> (B, D, H/P, W/P): one D-dim embedding per patch
        patches = self.patch_embedding(x)
        # (B, D, H/P, W/P) -> (B, D, N): N = (H/P)·(W/P) patches
        patches = patches.flatten(2)
        # (B, D, N) -> (B, N, D)
        patches = patches.transpose(1, 2)
        # (B, N, D) -> (B, N, D)
        patches = self.linear_embedding(patches)

        # (1, 1, D) -> (B, 1, D): the same [CLS] parameter for every image (a view, no copy)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # (B, 1, D) + (B, N, D) -> (B, N + 1, D)
        x = torch.cat([cls_token, patches], dim=1)

        # (B, N + 1, D) + positional encoding (broadcast over the batch), then dropout
        return self.positional_encoding(x)
