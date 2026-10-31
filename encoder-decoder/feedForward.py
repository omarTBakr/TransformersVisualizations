import torch
from torch import nn

from config import get_settings


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def get_activation(activation_name: str) -> nn.Module:
    """Return an activation module from its configuration name."""
    try:
        return _ACTIVATIONS[activation_name.lower()]()
    except KeyError:
        raise ValueError(
            f"Unsupported activation function: {activation_name}. "
            f"Choose one of {sorted(_ACTIVATIONS)}"
        ) from None


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> activation -> Dropout -> Linear."""

    def __init__(self):
        super().__init__()
        cfg = get_settings().model
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias),
            get_activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
