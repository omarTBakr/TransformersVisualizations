"""
Generate ui/architecture.html: an interactive Transformer diagram in which every block
links to the file / class in this repo that implements it, plus a step-by-step trace of a
real forward pass. The page shell and the generic code live in ../shared (see shared/py/build.py).

Usage:
    .venv/bin/python ui/build_ui.py

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

TITLE = "Transformer Architecture Map"
# Wording the shared stage engine uses for this model's output.
SPEC = {"out_axis": "vocabulary", "out_item": "next word", "top_label": "most likely next words",
        "top_title": "next-token probabilities"}


# ----------------------------------------------------------------------------------
# Diagram block  ->  implementing file / symbol.  `symbol` is a class name, a
# "Class.method" qualname, a top-level function name, or None for the whole file.
# ----------------------------------------------------------------------------------
COMPONENTS: list[Component] = [
    Component("input_embedding", "Input Embedding", "embed",
              "Looks up source token ids and scales them by √d_model.",
              [("inputEmbedding.py", "InputEmbedding")]),
    Component("output_embedding", "Output Embedding", "embed",
              "Looks up target token ids (shifted right) and scales them by √d_model. "
              "Its weight is tied to the final Linear layer when share_embeddings is on.",
              [("outputEmbedding.py", "OutputEmbedding")]),
    Component("positional_encoding", "Positional Encoding", "pe",
              "Adds sinusoidal or learned position information, then applies dropout. "
              "One instance is shared by the encoder and decoder inputs.",
              [("positionalEncoding.py", "PositionalEncoding")]),
    Component("encoder_stack", "Encoder (N× stack)", "stack",
              "Runs the input through N EncoderLayers, then a final LayerNorm.",
              [("encoder.py", "Encoder")]),
    Component("encoder_layer", "Encoder Layer", "stack",
              "One encoder block: self-attention → Add & Norm → feed-forward → Add & Norm.",
              [("encoder.py", "EncoderLayer")]),
    Component("enc_self_attn", "Multi-Head Attention", "attn",
              "Encoder self-attention: queries, keys and values all come from the source sequence. "
              "The source padding mask blocks attention to padding tokens.",
              [("multiheadAttention.py", "MultiHeadAttention"),
               ("multiheadAttention.py", "merge_masks")]),
    Component("enc_add_norm", "Add & Norm", "norm",
              "Residual connection plus post-LayerNorm (norm1 / norm2 in EncoderLayer).",
              [("encoder.py", "EncoderLayer.forward"), ("encoder.py", "EncoderLayer.__init__")]),
    Component("enc_ffn", "Feed Forward", "ffn",
              "Position-wise Linear → activation → Dropout → Linear.",
              [("feedForward.py", "FeedForward"), ("feedForward.py", "get_activation")]),
    Component("decoder_stack", "Decoder (N× stack)", "stack",
              "Runs the target through N DecoderLayers, then a final LayerNorm.",
              [("decoder.py", "Decoder")]),
    Component("decoder_layer", "Decoder Layer", "stack",
              "One decoder block: masked self-attention → cross-attention → feed-forward, "
              "with Add & Norm after each.",
              [("decoder.py", "DecoderLayer")]),
    Component("dec_masked_attn", "Masked Multi-Head Attention", "attn",
              "Decoder self-attention with a causal mask, so position i can only see positions ≤ i.",
              [("multiheadAttention.py", "MultiHeadAttention"),
               ("transformer.py", "Transformer.generate_square_subsequent_mask"),
               ("transformer.py", "Transformer.decode")]),
    Component("dec_add_norm", "Add & Norm", "norm",
              "Residual connection plus post-LayerNorm (norm1 / norm2 / norm3 in DecoderLayer).",
              [("decoder.py", "DecoderLayer.forward"), ("decoder.py", "DecoderLayer.__init__")]),
    Component("dec_cross_attn", "Multi-Head Attention", "attn",
              "Cross-attention: queries come from the decoder; keys and values come from the "
              "encoder output (memory).",
              [("multiheadAttention.py", "MultiHeadAttention"), ("decoder.py", "DecoderLayer.forward")]),
    Component("dec_ffn", "Feed Forward", "ffn",
              "Position-wise Linear → activation → Dropout → Linear (same module as in the encoder).",
              [("feedForward.py", "FeedForward")]),
    Component("linear", "Linear", "linear",
              "The `generator` projection from d_model to tgt_vocab_size; produces the logits.",
              [("transformer.py", "Transformer.__init__"), ("transformer.py", "Transformer.forward")]),
    Component("softmax", "Softmax", "softmax",
              "Not part of the model: Transformer.forward returns raw logits. Softmax is applied "
              "by the loss (cross-entropy) during training, or when decoding at inference time.",
              [("transformer.py", "Transformer.forward")]),
    Component("padding_mask", "Padding mask", "misc",
              "Builds (batch, seq) boolean masks where True marks padding tokens.",
              [("transformer.py", "Transformer.make_padding_mask")]),
    Component("transformer", "Transformer (model)", "misc",
              "Top-level module that wires the embeddings, encoder, decoder and output projection together.",
              [("transformer.py", "Transformer")]),
    Component("config", "Configuration", "misc",
              "All hyperparameters are read from config.yml and validated by the pydantic models in config.py.",
              [("config.yml", None), ("config.py", "ModelConfig")]),
    # Not on the diagram
    Component("self_attention", "SelfAttention (single head)", "misc",
              "A standalone single-head attention module. The model itself doesn't use it.",
              [("selfAttention.py", "SelfAttention")]),
    Component("main", "main.py (smoke test)", "misc",
              "Builds the model from config.yml and runs a forward pass on a dummy batch.",
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
        return {k: getattr(mm, k) for k in ("d_model", "n_heads", "n_encoder_layers", "n_decoder_layers", "d_ff",
                                             "dropout", "activation", "positional_encoding", "share_embeddings",
                                             "src_vocab_size", "tgt_vocab_size", "max_seq_len")}

    # Simulate every preset (tiny / small / full); the page switches between them.
    sims, configs, preset_info = {}, {}, []
    for pr in PRESETS:
        ps = build_settings(ROOT / "config.yml", pr)
        sims[pr.name] = simulate(ps, pr.src, pr.tgt, detail=pr.detail, src_bos=pr.src_bos)
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
        print(f"  {c['label']:<30} -> {refs}")
    for name, sm in sims.items():
        c = configs[name]
        print(f"\nPreset {name:5}: d_model {c['d_model']}, heads {c['n_heads']}, layers {c['n_encoder_layers']}+{c['n_decoder_layers']}"
              f" → {len(sm['steps'])} steps")
    print(f"Pinned {len(data['impl'])} implementation notes (details not in the paper)")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
