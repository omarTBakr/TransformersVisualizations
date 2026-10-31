"""
Generate ui/architecture.html: an interactive Transformer diagram in which every block
links to the file / class in this repo that implements it.

Usage:
    .venv/bin/python ui/build_ui.py

Re-run after changing the code so the page shows the current source.
"""

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
TEMPLATE = UI_DIR / "template.html"
OUTPUT = UI_DIR / "architecture.html"
PLACEHOLDER = "/*__DATA__*/null"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UI_DIR))
from config import Settings  # noqa: E402


@dataclass
class Component:
    id: str
    label: str
    kind: str  # embed | pe | attn | ffn | norm | linear | softmax | stack | misc
    note: str
    refs: list[tuple[str, str | None]] = field(default_factory=list)  # (file, symbol)


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


def find_symbol(tree: ast.Module, qualname: str) -> ast.AST:
    """Locate a class / function (optionally Class.method) in a parsed module."""
    node: ast.AST = tree
    for part in qualname.split("."):
        body = getattr(node, "body", [])
        match = next(
            (n for n in body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == part),
            None,
        )
        if match is None:
            raise LookupError(qualname)
        node = match
    return node


def resolve(file: str, symbol: str | None, cache: dict) -> dict:
    path = ROOT / file
    if not path.exists():
        raise SystemExit(f"[build_ui] File not found: {file}")
    text = cache.setdefault(file, path.read_text(encoding="utf-8"))
    lines = text.splitlines()

    if symbol is None:
        start, end = 1, len(lines)
    else:
        if not file.endswith(".py"):
            raise SystemExit(f"[build_ui] Symbols are only supported for .py files: {file}")
        try:
            node = find_symbol(ast.parse(text, filename=file), symbol)
        except LookupError:
            raise SystemExit(
                f"[build_ui] '{symbol}' not found in {file}. "
                "Update COMPONENTS in ui/build_ui.py to match the code."
            ) from None
        # Include decorators, if any.
        decos = getattr(node, "decorator_list", [])
        start = min([node.lineno] + [d.lineno for d in decos])
        end = node.end_lineno

    return {
        "file": file,
        "symbol": symbol,
        "start": start,
        "end": end,
        "code": "\n".join(lines[start - 1 : end]),
    }


def main() -> None:
    settings = Settings.from_yaml(ROOT / "config.yml")
    m = settings.model

    cache: dict[str, str] = {}
    components = {}
    for c in COMPONENTS:
        components[c.id] = {
            "label": c.label,
            "kind": c.kind,
            "note": c.note,
            "refs": [resolve(f, s, cache) for f, s in c.refs],
        }

    # Reverse mapping: every project source file -> the blocks it implements.
    project_files = sorted(p.name for p in ROOT.glob("*.py")) + ["config.yml"]
    files = {}
    for name in project_files:
        text = cache.get(name) or (ROOT / name).read_text(encoding="utf-8")
        files[name] = {
            "code": text,
            "lines": len(text.splitlines()),
            "blocks": [cid for cid, c in components.items()
                       if any(r["file"] == name for r in c["refs"])],
        }

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
    sim = sims["full"]
    paper = paper_data(settings)
    impl = resolve_notes(m.d_model)
    known = {c.id for c in COMPONENTS} | {s["key"] for s in sim["steps"]}
    bad = sorted({c for n in impl for c in n["comps"] if c not in known})
    if bad:
        raise SystemExit(f"[build_ui] Implementation notes reference unknown components: {bad}")
    missing = [c.id for c in COMPONENTS if c.id not in paper["entries"]]
    if missing:
        raise SystemExit(f"[build_ui] No paper entry for components: {missing} (see ui/paper.py)")

    data = {
        "sims": sims,
        "configs": configs,
        "presets": preset_info,
        "paper": paper,
        "impl": impl,
        "config": configs["full"],
        "components": components,
        "files": files,
    }

    # The page loads these with relative <script src> tags (works from file:// too).
    missing_js = [f for f in ("js/dock.js", "js/tensors.js", "js/stage.js") if not (UI_DIR / f).exists()]
    if missing_js:
        raise SystemExit(f"[build_ui] Missing UI scripts: {missing_js}")

    template = TEMPLATE.read_text(encoding="utf-8")
    # Cache-bust the local scripts so a rebuilt page never runs a stale copy from the browser cache.
    import hashlib  # noqa: E402
    for js in ("js/dock.js", "js/tensors.js", "js/stage.js"):
        ver = hashlib.md5((UI_DIR / js).read_bytes()).hexdigest()[:8]
        template = template.replace(f'src="{js}"', f'src="{js}?v={ver}"')
    if PLACEHOLDER not in template:
        raise SystemExit(f"[build_ui] Placeholder {PLACEHOLDER!r} missing from template.html")
    # Escape "</" so embedded source can never close the <script> tag.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUTPUT.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")

    for cid, c in components.items():
        refs = ", ".join(f"{r['file']}:{r['start']}-{r['end']}" for r in c["refs"])
        print(f"  {c['label']:<30} -> {refs}")
    for name, sm in sims.items():
        c = configs[name]
        print(f"\nPreset {name:5}: d_model {c['d_model']}, heads {c['n_heads']}, layers {c['n_encoder_layers']}+{c['n_decoder_layers']}"
              f" → {len(sm['steps'])} steps")
    print(f"Pinned {len(impl)} implementation notes (details not in the paper)")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
