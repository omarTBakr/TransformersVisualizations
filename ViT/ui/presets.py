"""
Model-size presets for the ViT page, each run on a small batch of synthetic images.

  tiny   6×6 grayscale, 3×3 patches → 4 patches + [CLS], D = 3, 1 head, 1 layer: every number fits on screen
  small  12×12 RGB, 4×4 patches → 9 patches + [CLS], D = 9, 3 heads, 2 layers
  full   the real config.yml (ViT-Base/16 at 224×224: 196 patches + [CLS], D = 768)

Each preset is simulated at build time; the page lets the user switch between them.
Only the UI uses these: the model code still reads its settings from config.yml.
"""

import math
from dataclasses import dataclass, field

import torch
import yaml

from config import Settings


@dataclass
class Preset:
    name: str
    label: str
    blurb: str
    classes: list[str] | None          # class names (None: "class k")
    images: str                        # which synthetic batch to draw (see make_images)
    detail: bool                       # record every value (no sampling) + matmul operands
    model: dict = field(default_factory=dict)


PRESETS = [
    Preset(
        name="tiny", label="Tiny 6×6",
        blurb="6×6 grayscale images cut into four 3×3 patches, D = 3, 1 head, 1 layer: every value is shown "
              "and every product can be checked by hand.",
        classes=["vertical", "horizontal", "diagonal"], images="lines", detail=True,
        model=dict(image_size=6, patch_size=3, in_channels=1, num_classes=3, d_model=3, n_heads=1, d_ff=6,
                   n_encoder_layers=1, max_seq_len=5),
    ),
    Preset(
        name="small", label="Small 12×12",
        blurb="12×12 RGB images cut into nine 4×4 patches, D = 9, 3 heads, 2 layers: still small enough to show every value.",
        classes=["square", "cross", "ring"], images="shapes", detail=True,
        model=dict(image_size=12, patch_size=4, in_channels=3, num_classes=3, d_model=9, n_heads=3, d_ff=18,
                   n_encoder_layers=2, max_seq_len=10),
    ),
    Preset(
        name="full", label="ViT-Base/16 (config.yml)",
        blurb="The real configuration from config.yml: 224×224 RGB, 196 patches + [CLS], D = 768, 12 heads, 12 layers. "
              "Feature axes are sampled (16 dims) and sequences show their first 16 tokens.",
        classes=None, images="scene", detail=False,
    ),
]


def build_settings(base_yaml, preset: Preset) -> Settings:
    """config.yml with the preset's model overrides."""
    with open(base_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if preset.model:
        data = {**data, "model": {**data["model"], **preset.model}}
    return Settings.model_validate(data)


def make_images(kind: str, size: int, channels: int) -> tuple[torch.Tensor, list[int | None]]:
    """A deterministic batch of 2 synthetic images in [0, 1], shape (2, C, H, W), and their true classes."""
    H = W = size
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float), torch.arange(W, dtype=torch.float), indexing="ij")
    if kind == "lines":                       # grayscale: a vertical bar and a diagonal
        a = torch.full((H, W), 0.1)
        a[:, 2] = 0.9
        b = torch.full((H, W), 0.1)
        b[yy == xx] = 0.9
        return torch.stack([a, b])[:, None], [0, 2]
    if kind == "shapes":                      # RGB: a red square and a blue ring
        a = torch.zeros(3, H, W) + 0.05
        sq = (yy >= 3) & (yy <= 8) & (xx >= 3) & (xx <= 8)
        a[0][sq], a[1][sq], a[2][sq] = 0.9, 0.2, 0.1
        b = torch.zeros(3, H, W) + 0.05
        r = ((yy - 5.5) ** 2 + (xx - 5.5) ** 2).sqrt()
        ring = (r >= 3) & (r <= 4.6)
        b[0][ring], b[1][ring], b[2][ring] = 0.1, 0.3, 0.9
        return torch.stack([a, b]), [0, 2]
    # "scene": a 224×224 sky with a sun over green ground, and a checkerboard
    t = yy / (H - 1)
    a = torch.stack([0.35 + 0.3 * t, 0.55 + 0.25 * t, 0.95 - 0.2 * t])
    sun = ((yy - 60) ** 2 + (xx - 150) ** 2) < 28 ** 2
    a[0][sun], a[1][sun], a[2][sun] = 1.0, 0.85, 0.2
    ground = yy > 160
    a[0][ground], a[1][ground], a[2][ground] = 0.2, 0.6, 0.25
    chk = ((yy // 28 + xx // 28) % 2).float()
    b = torch.stack([0.15 + 0.7 * chk, 0.15 + 0.7 * chk, 0.15 + 0.7 * chk])
    return torch.stack([a, b])[:, :channels], [None, None]


def downsample(img: torch.Tensor, max_side: int = 56) -> tuple[torch.Tensor, int]:
    """Nearest-neighbour thumbnail of a (C, H, W) image for the page, and the step used."""
    stride = max(1, math.ceil(img.shape[-1] / max_side))
    return img[:, ::stride, ::stride], stride
