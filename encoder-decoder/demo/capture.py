"""
Regenerate the demo assets from the live app:

  demo/screenshots/*.png      feature screenshots (headless Chrome, 1920×1080)
  demo/linkedin-carousel.pdf  3-page LinkedIn document post (1080×1350)
  demo/walkthrough.mp4        ~1-minute captioned tour of all features (+ walkthrough.gif for the README)

Usage (from encoder-decoder/):
    uv run ui/build_ui.py          # make sure the page is up to date
    uv run demo/capture.py         # needs google-chrome and ffmpeg on PATH

The shared machinery is in shared/py/capture.py. The paper's page viewer is deliberately not captured: full pages of the paper go beyond
the tables-and-figures permission, so they stay out of public demo material.
"""

import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO.parent.parent / "shared" / "py"))
import capture  # noqa: E402  (shared/py/capture.py)

# (file name, query string, step, caption)
SHOTS_SPEC = [
    ("01-overview", "config=tiny&stage=1", 5, "One page: the operation, the architecture and the code that runs it, all in sync"),
    ("02-embedding", "config=tiny&stage=2", 3, "Every token is looked up in a real embedding table"),
    ("03-positional-encoding", "config=tiny&stage=2", 4, "Positional encoding, position by position, with the sin / cos formula worked out"),
    ("04-matmul", "config=tiny&stage=1", 5, "Matrix multiplications: pick any cell to see its dot product, checked against the model"),
    ("05-attention-scores", "config=tiny&stage=5", 5, "Attention scores Q·Kᵀ / √d_k with the real numbers"),
    ("06-softmax", "config=tiny&sample=2&stage=7", 5, "Softmax written out in LaTeX; padded keys get −∞ and weight 0"),
    ("07-layernorm", "config=tiny&stage=4", 6, "LayerNorm: mean and variance along the feature axis, step by step"),
    ("08-relu", "config=tiny&stage=2", 7, "ReLU: negatives flagged, zeros in mint, the max starred"),
    ("09-small-heads", "config=small&stage=4", 5, "Small 9×9 preset: 3 attention heads, 2 + 2 layers, padding"),
    ("10-full-cubes", "config=full&stage=1", 4, "Full config.yml (d_model 512): isometric tensors with sampled values"),
    ("11-output-softmax", "config=tiny&stage=4", 21, "Next-token probabilities, with every exp and Σ shown"),
    ("12-read-the-code", "config=tiny&stage=6", 5, "Read the code: each stage outlines its exact line, plus details not in the paper"),
]


if __name__ == "__main__":
    capture.run(DEMO, SHOTS_SPEC)
