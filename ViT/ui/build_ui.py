"""
Generate ui/architecture.html for the Vision Transformer: an interactive diagram in which every
block links to the file / class in this repo that implements it, plus a step-by-step trace of a
real forward pass on synthetic images. The page shell and the generic code live in ../shared.

Usage:
    uv run ui/build_ui.py

Re-run after changing the code so the page shows the current source.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(ROOT.parent / "shared" / "py"))
from config import Settings  # noqa: E402
from build import Component, components_and_files, check, write_page  # noqa: E402

TITLE = "Vision Transformer (ViT) Map"
# Wording the shared stage engine uses for this model.
SPEC = {"out_axis": "classes", "out_item": "class", "top_label": "most likely classes",
        "top_title": "class probabilities", "ffn_title": "MLP (feed-forward)", "ffn_name": "MLP"}

# ----------------------------------------------------------------------------------
# Diagram block  ->  implementing file / symbol.
# ----------------------------------------------------------------------------------
COMPONENTS: list[Component] = [
    Component("patch_embedding", "Linear Projection of Flattened Patches", "embed",
              "Cuts the image into P×P patches and maps each to D numbers (the paper's E), computed as a Conv2d "
              "with kernel = stride = P.",
              [("ImageEmbedding.py", "ImageEmbedding.forward"), ("ImageEmbedding.py", "ImageEmbedding.__init__")]),
    Component("extra_linear", "Extra Linear (not in the paper)", "linear",
              "A Linear D → D applied to every patch embedding. Not in the paper: it composes with E into one "
              "linear map, so it adds parameters but no expressive power.",
              [("ImageEmbedding.py", "ImageEmbedding.forward"), ("ImageEmbedding.py", "ImageEmbedding.__init__")]),
    Component("cls_token", "[class] token", "embed",
              "A learned vector prepended to the patch sequence; its final state represents the whole image.",
              [("ImageEmbedding.py", "ImageEmbedding.forward"), ("ImageEmbedding.py", "ImageEmbedding.__init__")]),
    Component("positional_encoding", "Position Embedding", "pe",
              "Adds one D-dim position vector per token, then dropout. Sinusoidal here (config.yml); the paper "
              "learns them.",
              [("positionalEncoding.py", "PositionalEncoding")]),
    Component("encoder_stack", "Transformer Encoder (L× stack)", "stack",
              "Runs the tokens through L pre-norm encoder layers, then a final LayerNorm (Eq. 4).",
              [("encoder.py", "Encoder")]),
    Component("encoder_layer", "Encoder Layer", "stack",
              "Norm → multi-head self-attention → + residual → Norm → MLP → + residual (Eq. 2–3).",
              [("encoder.py", "EncoderLayer")]),
    Component("enc_norm", "Norm", "norm",
              "Pre-norm LayerNorm before each block (norm1 / norm2 in EncoderLayer).",
              [("encoder.py", "EncoderLayer.forward"), ("encoder.py", "EncoderLayer.__init__")]),
    Component("enc_self_attn", "Multi-Head Attention", "attn",
              "Self-attention over all N + 1 tokens, with no masks.",
              [("multiheadAttention.py", "MultiHeadAttention")]),
    Component("enc_residual", "+ (residual)", "norm",
              "Adds the block's output back to its un-normalized input.",
              [("encoder.py", "EncoderLayer.forward")]),
    Component("enc_ffn", "MLP", "ffn",
              "Per-token Linear → GELU → Dropout → Linear.",
              [("feedForward.py", "FeedForward"), ("feedForward.py", "get_activation")]),
    Component("cls_select", "Take [CLS]", "misc",
              "Keeps token 0 of every image: the [class] token's final representation y.",
              [("transformer.py", "VisionTransformer.forward")]),
    Component("head", "MLP Head", "linear",
              "The classification head: one Linear D → K (the paper's fine-tuning head); produces the logits.",
              [("transformer.py", "VisionTransformer.__init__"), ("transformer.py", "VisionTransformer.forward")]),
    Component("softmax", "Class", "softmax",
              "Not part of the model: VisionTransformer.forward returns logits. Softmax turns them into class "
              "probabilities (inside the cross-entropy loss during training).",
              [("transformer.py", "VisionTransformer.forward")]),
    Component("vit", "VisionTransformer (model)", "misc",
              "Top-level module: image embedding → encoder → [CLS] → head.",
              [("transformer.py", "VisionTransformer")]),
    Component("image_embedding", "ImageEmbedding (Eq. 1)", "misc",
              "Patches → projection → extra Linear → [CLS] → position embedding: all of Eq. 1 in one module.",
              [("ImageEmbedding.py", "ImageEmbedding")]),
    Component("config", "Configuration", "misc",
              "All hyperparameters are read from config.yml and validated by the pydantic models in config.py.",
              [("config.yml", None), ("config.py", "ModelConfig")]),
    Component("main", "main.py (smoke test)", "misc",
              "Builds the model from config.yml and runs a forward pass on random images.",
              [("main.py", "main"), ("main.py", "resolve_device")]),
]


def main() -> None:
    settings = Settings.from_yaml(ROOT / "config.yml")
    m = settings.model
    components, files = components_and_files(ROOT, COMPONENTS)

    from simulate import simulate  # noqa: E402  (needs sys.path set above)
    from paper import paper_data  # noqa: E402
    from impl_notes import resolve_notes  # noqa: E402
    from presets import PRESETS, build_settings  # noqa: E402

    def cfg_dict(mm) -> dict:
        d = {k: getattr(mm, k) for k in ("image_size", "patch_size", "in_channels", "num_classes", "d_model", "n_heads",
                                          "n_encoder_layers", "d_ff", "dropout", "activation", "positional_encoding",
                                          "max_seq_len")}
        return d | {"n_patches": mm.n_patches}

    sims, configs, preset_info = {}, {}, []
    for pr in PRESETS:
        ps = build_settings(ROOT / "config.yml", pr)
        sims[pr.name] = simulate(ps, pr.classes, pr.images, detail=pr.detail)
        configs[pr.name] = cfg_dict(ps.model)
        preset_info.append({"name": pr.name, "label": pr.label, "blurb": pr.blurb, "detail": pr.detail})

    data = {
        "sims": sims,
        "configs": configs,
        "presets": preset_info,
        "paper": paper_data(settings),
        "impl": resolve_notes(m.d_model),
        "config": configs["full"],
        "components": components,
        "files": files,
        "spec": SPEC,
    }
    check(data, COMPONENTS)
    out = write_page(UI_DIR, data, TITLE)

    for cid, c in components.items():
        refs = ", ".join(f"{r['file']}:{r['start']}-{r['end']}" for r in c["refs"])
        print(f"  {c['label']:<40} -> {refs}")
    for name, sm in sims.items():
        c = configs[name]
        print(f"\nPreset {name:5}: {c['image_size']}² × {c['in_channels']}, P {c['patch_size']} → {c['n_patches']} patches, "
              f"D {c['d_model']}, heads {c['n_heads']}, layers {c['n_encoder_layers']} → {len(sm['steps'])} steps")
    print(f"Pinned {len(data['impl'])} implementation notes (details not in the paper)")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
