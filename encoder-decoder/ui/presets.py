"""
Model-size presets for the UI.

  tiny   3×3   d_model 3, 1 head, 1 + 1 layers, ≤ 3 tokens: every number fits on screen
  small  9×9   d_model 9, 3 heads, 2 + 2 layers, ≤ 9 tokens with padding
  full         the real config.yml

Each preset is simulated at build time; the page lets the user switch between them.
Only the UI uses these: the model code still reads its settings from config.yml.
"""

from dataclasses import dataclass, field

import yaml

from config import Settings

SPECIAL_TOKENS = 4  # <pad>, <unk>, <s>, </s>


@dataclass
class Preset:
    name: str
    label: str
    blurb: str
    src: list[str]
    tgt: list[str]
    detail: bool                       # record every value (no sampling) + matmul operands
    src_bos: bool = True               # start the encoder input with <s>
    model: dict = field(default_factory=dict)


def _vocab_size(sentences: list[str]) -> int:
    return SPECIAL_TOKENS + len({w for s in sentences for w in s.split()})


PRESETS = [
    Preset(
        name="tiny", label="Tiny 3×3",
        blurb="d_model 3, 1 head, 1 encoder + 1 decoder layer: every value is shown and every product can be checked by hand. "
              "The encoder input has no <s> here, so the shorter sentence is padded within 3 tokens.",
        src=["cat sleeps", "hi"], tgt=["Katze", "hallo"], detail=True, src_bos=False,
        model=dict(d_model=3, n_heads=1, d_ff=6, n_encoder_layers=1, n_decoder_layers=1, max_seq_len=8),
    ),
    Preset(
        name="small", label="Small 9×9",
        blurb="d_model 9, 3 heads, 2 encoder + 2 decoder layers, with padding: still small enough to show every value.",
        src=["the cat sat on the mat", "I love you"], tgt=["die Katze saß auf der Matte", "ich liebe dich"], detail=True,
        model=dict(d_model=9, n_heads=3, d_ff=18, n_encoder_layers=2, n_decoder_layers=2, max_seq_len=16),
    ),
    Preset(
        name="full", label="Full (config.yml)",
        blurb="The real configuration from config.yml. Feature axes are sampled (16 dims shown).",
        src=["the cat sat on the mat", "I love you"], tgt=["die Katze saß auf der Matte", "ich liebe dich"], detail=False,
    ),
]


def build_settings(base_yaml: str, preset: Preset) -> Settings:
    """config.yml with the preset's model overrides (vocab sizes sized to the toy vocabulary)."""
    with open(base_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if preset.model:
        model = {**data["model"], **preset.model}
        model["src_vocab_size"] = max(model.get("src_vocab_size", 0) if not preset.detail else 0, _vocab_size(preset.src))
        model["tgt_vocab_size"] = max(model.get("tgt_vocab_size", 0) if not preset.detail else 0, _vocab_size(preset.tgt), 5)
        model["vocab_size"] = max(model["src_vocab_size"], model["tgt_vocab_size"])
        data = {**data, "model": model}
    return Settings.model_validate(data)
