"""
Regenerate the ViT demo assets from the live page:

  demo/screenshots/*.png      feature screenshots (headless Chrome, 1920×1080)
  demo/linkedin-carousel.pdf  3-page LinkedIn document post (1080×1350)
  demo/walkthrough.mp4        captioned tour of the features (+ walkthrough.gif for the README)

Usage (from ViT/):
    uv run ui/build_ui.py          # make sure the page is up to date
    uv run demo/capture.py         # needs google-chrome and ffmpeg on PATH

The shared machinery is in shared/py/capture.py. The paper's page viewer is never captured:
the ViT paper carries no reproduction permission, so its pages stay out of demo material.
"""

import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO.parent.parent / "shared" / "py"))
import capture  # noqa: E402  (shared/py/capture.py)

# (file name, query string, step, caption)
SHOTS_SPEC = [
    ("01-overview", "config=tiny&stage=5", 6, "One page: the operation, the architecture and the code that runs it, all in sync"),
    ("02-patches", "config=small&stage=2", 1, "The image is cut into P×P patches, numbered in raster order"),
    ("03-flatten", "config=tiny&stage=3", 1, "Each patch becomes one row of P²·C pixel values"),
    ("04-conv-matmul", "config=tiny&stage=7", 1, "The Conv2d patch projection is the matmul x_p·E: pick any cell, checked against the model"),
    ("05-extra-linear", "config=tiny&stage=2", 2, "The extra Linear (not in the paper) composes with E into one linear map"),
    ("06-cls-token", "config=tiny&stage=2", 3, "The learned [class] token is prepended at position 0"),
    ("07-position", "config=tiny&stage=2", 4, "Position embeddings, row by row (sinusoidal here, learned in the paper)"),
    ("08-prenorm", "config=tiny&stage=2", 5, "Pre-norm LayerNorm: mean and variance of each token"),
    ("09-residual", "config=tiny&stage=1", 7, "The residual adds back the un-normalized input"),
    ("10-gelu", "config=tiny&stage=2", 9, "GELU x·Φ(x), worked out with the error function"),
    ("11-vit-base", "config=full&stage=2", 1, "ViT-Base/16: a 224×224 image, 196 patches of 16×16"),
    ("12-classify", "config=tiny&stage=4", 14, "Take [CLS], apply the head, softmax over the classes"),
    ("13-read-the-code", "config=tiny&stage=8", 1, "Read the code: each stage outlines its exact line, plus details not in the paper"),
]


if __name__ == "__main__":
    capture.run(DEMO, SHOTS_SPEC)
