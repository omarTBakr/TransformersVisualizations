import torch
from torch import nn

from config import get_settings
from encoder import Encoder
from ImageEmbedding import ImageEmbedding


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification. All hyperparameters come from config.yml.

        images → patch embeddings + [CLS] + positional encoding   (Eq. 1)
               → L pre-norm encoder layers                         (Eq. 2–3)
               → final LayerNorm, keep the [CLS] token             (Eq. 4: y = LN(z_L⁰))
               → linear classification head → class logits
    """

    def __init__(self):
        super().__init__()
        cfg = get_settings().model

        self.image_embedding = ImageEmbedding()
        self.encoder = Encoder()
        # Classification head: the paper uses one Linear layer when fine-tuning
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform init for weight matrices (2-D and conv kernels); [CLS] keeps its own init."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "cls_token" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, in_channels, image_size, image_size)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        x = self.image_embedding(images)
        x = self.encoder(x)
        cls_output = x[:, 0, :]            # the [CLS] token's final representation
        return self.head(cls_output)
