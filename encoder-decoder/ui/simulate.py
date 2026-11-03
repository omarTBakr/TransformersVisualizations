"""
Run one real forward pass of the encoder–decoder Transformer on a small simulated batch and
record a step-by-step execution trace for the UI (the generic tracing lives in shared/py/tracer.py).
"""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "shared" / "py"))
import tracer as tr  # noqa: E402
from tracer import Tracer, DimNamer, cube_slice, fmt as _fmt, rl as _rl, shape as _shape  # noqa: E402

# Default simulated translation batch (en → de). Sample 2 is shorter, so it gets padded.
SRC_SENTENCES = ["the cat sat on the mat", "I love you"]
TGT_SENTENCES = ["die Katze saß auf der Matte", "ich liebe dich"]

# Project modules that read the configuration through `get_settings()`.
SETTINGS_MODULES = ["transformer", "encoder", "decoder", "multiheadAttention", "feedForward",
                    "inputEmbedding", "outputEmbedding", "positionalEncoding", "selfAttention"]

# ----------------------------------------------------------------------------------
# Step metadata: trace key -> (diagram block key, component id, tensor role)
# ----------------------------------------------------------------------------------
STEP_MAP = {
    "src_pad_mask": ("padding_mask", "padding_mask", "src"),
    "tgt_pad_mask": ("padding_mask", "padding_mask", "tgt"),
    "input_embedding": ("input_embedding", "input_embedding", "src"),
    "pe_enc": ("pe_enc", "positional_encoding", "src"),
    "enc_self_attn": ("enc_self_attn", "enc_self_attn", "src"),
    "enc_norm1": ("enc_norm1", "enc_add_norm", "src"),
    "enc_ffn": ("enc_ffn", "enc_ffn", "src"),
    "enc_norm2": ("enc_norm2", "enc_add_norm", "src"),
    "enc_final": ("encoder_stack", "encoder_stack", "src"),
    "causal_mask": ("dec_masked_attn", "dec_masked_attn", "tgt"),
    "output_embedding": ("output_embedding", "output_embedding", "tgt"),
    "pe_dec": ("pe_dec", "positional_encoding", "tgt"),
    "dec_masked_attn": ("dec_masked_attn", "dec_masked_attn", "tgt"),
    "dec_norm1": ("dec_norm1", "dec_add_norm", "tgt"),
    "dec_cross_attn": ("dec_cross_attn", "dec_cross_attn", "both"),
    "dec_norm2": ("dec_norm2", "dec_add_norm", "tgt"),
    "dec_ffn": ("dec_ffn", "dec_ffn", "tgt"),
    "dec_norm3": ("dec_norm3", "dec_add_norm", "tgt"),
    "dec_final": ("decoder_stack", "decoder_stack", "tgt"),
    "generator": ("linear", "linear", "tgt"),
    "softmax": ("softmax", "softmax", "tgt"),
}


# Names for the operations a line performs (first matching pattern wins).
LABEL_RULES = [
        ("torch.matmul(q, k.transpose", "Dot products Q·Kᵀ, scaled by 1/√d_k"),
        ("torch.matmul(attn_weights", "Weighted sum of values: weights · V"),
        ("torch.softmax", "Softmax over the keys"),
        ("nan_to_num", "Zero any fully-masked rows"),
        ("scores + mask", "Add the mask: blocked scores become −∞"),
        ("x.view(batch_size, seq_len, self.n_heads", "Split into heads (view + transpose)"),
        (".contiguous().view", "Concatenate the heads (transpose + view)"),
        ("masked_fill(", "Boolean mask → 0 / −∞"),
        ("[:, None, None, :]", "Reshape padding mask to (B, 1, 1, S) so it broadcasts"),
        ("merged + kpm", "Merge causal and padding masks"),
        ("torch.triu", "Upper triangle above the diagonal (future positions)"),
        ("tokens == pad_token_id", "Compare every token with the pad id"),
        ("self.pe[:, :seq_len]", "Add the positional encoding"),
        ("* self.scale", "Scale by √d_model"),
    ]
_label = tr.make_labeler(LABEL_RULES)


def simulate(settings, src_sentences=None, tgt_sentences=None, detail: bool = False, src_bos: bool = True) -> dict:
    """Run one traced forward pass with `settings` (the project modules are pointed at them)."""
    with tr.detail_mode(detail), tr.use_settings(settings, SETTINGS_MODULES):
        return _simulate(settings, src_sentences or SRC_SENTENCES, tgt_sentences or TGT_SENTENCES, src_bos)


def _simulate(settings, SRC_SENTENCES, TGT_SENTENCES, src_bos: bool = True) -> dict:
    from transformer import Transformer  # imported lazily: needs sys.path set by caller

    m, tok = settings.model, settings.tokenizer
    torch.manual_seed(settings.training.seed)

    # ---- simulated batch -------------------------------------------------------
    # A toy word-level vocabulary stands in for the real tokenizer: the special tokens keep
    # their ids from config.yml and every new word gets the next free id.
    specials = {tok.pad_token_id: tok.pad_token, tok.unk_token_id: tok.unk_token,
                tok.bos_token_id: tok.bos_token, tok.eos_token_id: tok.eos_token}

    def build_vocab(sentences: list[str], size: int) -> dict[str, int]:
        vocab, next_id = {}, max(specials) + 1
        for word in (w for sent in sentences for w in sent.split()):
            if word not in vocab:
                vocab[word] = next_id
                next_id += 1
        if next_id > size:
            raise SystemExit("[build_ui] Toy vocabulary is larger than the configured vocab size")
        return vocab

    src_vocab = build_vocab(SRC_SENTENCES, m.src_vocab_size)
    tgt_vocab = build_vocab(TGT_SENTENCES, m.tgt_vocab_size)

    def pad_to(seqs: list[list[int]]) -> list[list[int]]:
        n = max(map(len, seqs))
        return [s + [tok.pad_token_id] * (n - len(s)) for s in seqs]

    # Encoder input:  <s> words </s> <pad>…
    src_ids = pad_to([([tok.bos_token_id] if src_bos else []) + [src_vocab[w] for w in s.split()] + [tok.eos_token_id]
                      for s in SRC_SENTENCES])
    # Teacher forcing. </s> is only ever an output of the decoder, never an input:
    #   decoder input = <s> words <pad>…   (shifted right)
    #   labels        = words </s> <pad>…  (what each position must predict)
    tgt_words = [[tgt_vocab[w] for w in s.split()] for s in TGT_SENTENCES]
    tgt_ids = pad_to([[tok.bos_token_id] + w for w in tgt_words])
    label_ids = pad_to([w + [tok.eos_token_id] for w in tgt_words])
    src, tgt = torch.tensor(src_ids), torch.tensor(tgt_ids)

    model = Transformer().eval()  # eval(): dropout disabled, so the trace is deterministic
    namer = DimNamer(src.size(0), [src.size(1), tgt.size(1)], m.d_model, m.d_ff, m.n_heads, m.tgt_vocab_size)
    tracer = Tracer(ROOT, namer, _label)
    tr.install_leaf_hooks(tracer, model)
    pe_calls = {"n": 0}

    # ---- forward hooks on the modules that correspond to diagram blocks ---------
    def pe_key(mod):                                     # one PositionalEncoding serves both sides
        pe_calls["n"] += 1
        return "pe_enc" if pe_calls["n"] == 1 else "pe_dec"

    def on_post(step, mod, args, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(mod, torch.nn.LayerNorm):
            tr.layernorm_subs(tracer, step, mod, args[0], out, post_norm=True)
        if isinstance(output, tuple) and len(output) > 1:
            # attention weights (B, H, T, S) -> average over heads per sample
            step["attn"] = output[1].mean(dim=1).round(decimals=3).tolist()

    def hook(module, key, layer=None):
        tracer.hook(module, pe_key if key == "pe" else key, layer, on_post)

    hook(model.input_embedding, "input_embedding")
    hook(model.positional_encoding, "pe")
    for i, layer in enumerate(model.encoder.layers, start=1):
        hook(layer.self_attn, "enc_self_attn", i)
        hook(layer.norm1, "enc_norm1", i)
        hook(layer.ffn, "enc_ffn", i)
        hook(layer.norm2, "enc_norm2", i)
    hook(model.encoder.norm, "enc_final")
    hook(model.output_embedding, "output_embedding")
    for i, layer in enumerate(model.decoder.layers, start=1):
        hook(layer.self_attn, "dec_masked_attn", i)
        hook(layer.norm1, "dec_norm1", i)
        hook(layer.cross_attn, "dec_cross_attn", i)
        hook(layer.norm2, "dec_norm2", i)
        hook(layer.ffn, "dec_ffn", i)
        hook(layer.norm3, "dec_norm3", i)
    hook(model.decoder.norm, "dec_final")
    hook(model.generator, "generator")

    # ---- wrap the static mask helpers so they become steps too ------------------
    orig_causal = Transformer.generate_square_subsequent_mask
    orig_pad = Transformer.make_padding_mask

    def traced_causal(size, device=None):
        tracer.begin("causal_mask", None, [])
        tracer.stack[-1]["scalar_inputs"] = [{"name": "size", "value": size}]
        mask = orig_causal(size, device=device)
        step = tracer.end(mask)
        step["out_shape"] = _shape(mask)
        step["out_dtype"] = "bool"
        step["mask"] = mask.int().tolist()
        return mask

    def traced_pad(key):
        def inner(tokens, pad_token_id):
            tracer.begin(key, None, [("tokens", tokens)])
            mask = orig_pad(tokens, pad_token_id)
            step = tracer.end(mask)
            step["out_shape"] = _shape(mask)
            step["out_dtype"] = "bool"
            step["mask"] = mask.int().tolist()
            return mask
        return inner

    Transformer.generate_square_subsequent_mask = staticmethod(traced_causal)
    try:
        sys.settrace(tracer.global_trace)
        with torch.no_grad():
            src_pad = traced_pad("src_pad_mask")(src, tok.pad_token_id)
            tgt_pad = traced_pad("tgt_pad_mask")(tgt, tok.pad_token_id)
            logits = model(src, tgt, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
    finally:
        sys.settrace(None)
        Transformer.generate_square_subsequent_mask = staticmethod(orig_causal)

    # ---- softmax (applied outside the model) ------------------------------------
    probs = torch.softmax(logits, dim=-1)
    last = [int(tgt_pad[b].logical_not().sum()) - 1 for b in range(tgt.size(0))]
    topk = []
    for b, pos in enumerate(last):
        p, idx = probs[b, pos].topk(min(5, probs.size(-1)))
        topk.append({"pos": pos, "ids": idx.tolist(), "probs": [round(v, 5) for v in p.tolist()]})
    gen_step = next(s for s in tracer.steps if s["key"] == "generator")
    lp, li = logits[0, last[0]].topk(min(5, logits.size(-1)))
    gen_step["topk_logits"] = {"pos": last[0], "ids": li.tolist(), "logits": [round(v, 3) for v in lp.tolist()]}
    sm_step = {"key": "softmax", "layer": None, "subs": []}
    tracer.stack.append(sm_step)
    mx = logits.max(dim=-1, keepdim=True).values
    ex = torch.exp(logits - mx)
    tot = ex.sum(dim=-1, keepdim=True)
    tracer.record_sub(mx, None, None, "Max logit per position (for numerical stability)", var="max", formula="m = max(z)")
    tracer.record_sub(ex, None, None, "Exponentiate the shifted logits", var="e", formula="e = exp(z − m)")
    tracer.record_sub(tot, None, None, "Sum over the vocabulary", var="Σ", formula="Σ = sum(e)")
    tracer.record_sub(probs, None, None, "Divide: probabilities that sum to 1", var="p", formula="p = e / Σ")
    tracer.stack.pop()
    tracer.steps.append({
        "subs": sm_step["subs"],
        "key": "softmax", "layer": None, "frames": [], "exec": {},
        "inputs": [{"name": "logits", "shape": list(logits.shape), "dtype": "float32"}],
        "in_shapes": [list(logits.shape)], "out_shape": list(probs.shape), "out_dtype": "float32",
        "topk": topk, "in_vals": cube_slice(logits), "out_vals": cube_slice(probs),
    })

    # ---- finalize: titles, descriptions, primary file ---------------------------
    d, h, V = m.d_model, m.n_heads, m.tgt_vocab_size
    nE, nD = m.n_encoder_layers, m.n_decoder_layers
    out = tr.finalize(tracer.steps, STEP_MAP,
                      lambda key, L, ins, o: _describe(key, L, nE, nD, ins, o, d, h, m, tok, V),
                      lambda key: nE if key.startswith("enc") else nD if key.startswith("dec") else None)

    return {
        "batch": {
            "src": src.tolist(), "tgt": tgt.tolist(), "labels": label_ids,
            "src_text": SRC_SENTENCES, "tgt_text": TGT_SENTENCES,
            "src_lang": settings.data.src_lang, "tgt_lang": settings.data.tgt_lang,
            # id -> word, for labelling blocks, heatmaps and predictions
            "src_vocab": {**{str(i): w for w, i in src_vocab.items()}, **{str(i): t for i, t in specials.items()}},
            "tgt_vocab": {**{str(i): w for w, i in tgt_vocab.items()}, **{str(i): t for i, t in specials.items()}},
            "pad": tok.pad_token_id, "bos": tok.bos_token_id, "eos": tok.eos_token_id,
        },
        "steps": out,
    }


def _describe(key, L, nE, nD, ins, o, d, h, m, tok, V):
    s_in = _fmt(ins[0]) if ins and ins[0] else ""
    s_out = _fmt(o) if o else ""
    hd = d // h
    enc = f"Encoder layer {L}/{nE} · " if L and key.startswith("enc") else ""
    dec = f"Decoder layer {L}/{nD} · " if L and key.startswith("dec") else ""
    table = {
        "src_pad_mask": ("Build source padding mask",
                         f"True where src == pad_token_id ({tok.pad_token_id}). Shape {s_out}. "
                         "Attention will give these key positions zero weight."),
        "tgt_pad_mask": ("Build target padding mask",
                         f"True where tgt == pad_token_id ({tok.pad_token_id}). Shape {s_out}."),
        "input_embedding": ("Input Embedding",
                            f"Look up each source token id {s_in} in a {m.src_vocab_size}×{d} table and "
                            f"scale by √{d} ≈ {math.sqrt(d):.1f} → {s_out}."),
        "pe_enc": ("Positional Encoding (encoder input)",
                   f"Add {m.positional_encoding} position vectors for positions 0–{ins[0][1] - 1 if ins[0] else '?'}, "
                   f"then dropout (off in eval mode). Shape stays {s_out}."),
        "enc_self_attn": (enc + "Multi-Head Self-Attention",
                          f"Q, K and V are projections of the same {s_in} tensor, split into {h} heads × {hd} dims. "
                          "Padded keys get −∞ before softmax, so they receive zero weight."),
        "enc_norm1": (enc + "Add & Norm (after attention)", "x + Dropout(attention output), then LayerNorm (norm1)."),
        "enc_ffn": (enc + "Feed Forward",
                    f"Linear {d}→{m.d_ff}, {m.activation}, Dropout, Linear {m.d_ff}→{d}, applied to every "
                    "position independently."),
        "enc_norm2": (enc + "Add & Norm (after FFN)", "x + Dropout(FFN output), then LayerNorm (norm2)."),
        "enc_final": ("Encoder output (final LayerNorm)",
                      f"The {s_out} result is the memory that every decoder layer's cross-attention reads."),
        "causal_mask": ("Build causal mask",
                        f"{s_out} upper-triangular boolean mask: position i may not attend to any j > i."),
        "output_embedding": ("Output Embedding",
                             f"Look up the shifted-right target ids {s_in} in a {V}×{d} table and scale by √{d} → {s_out}."),
        "pe_dec": ("Positional Encoding (decoder input)",
                   f"The same PositionalEncoding module, now applied to the target embeddings {s_out}."),
        "dec_masked_attn": (dec + "Masked Multi-Head Self-Attention",
                            f"Self-attention over the target {s_in} with the causal mask and the target "
                            "padding mask, so each position only sees earlier, non-pad positions."),
        "dec_norm1": (dec + "Add & Norm (after masked attention)", "x + Dropout(self-attention output), then LayerNorm (norm1)."),
        "dec_cross_attn": (dec + "Cross-Attention",
                           f"Queries come from the decoder {s_in}; keys and values come from the encoder memory "
                           f"{_fmt(ins[1]) if len(ins) > 1 else ''}. Source padding is masked."),
        "dec_norm2": (dec + "Add & Norm (after cross-attention)", "x + Dropout(cross-attention output), then LayerNorm (norm2)."),
        "dec_ffn": (dec + "Feed Forward",
                    f"Linear {d}→{m.d_ff}, {m.activation}, Dropout, Linear {m.d_ff}→{d}, per position."),
        "dec_norm3": (dec + "Add & Norm (after FFN)", "x + Dropout(FFN output), then LayerNorm (norm3)."),
        "dec_final": ("Decoder output (final LayerNorm)", f"Final decoder hidden states {s_out}."),
        "generator": ("Linear (generator)",
                      f"Project {d} → {V} logits for every target position: {s_in} → {s_out}."
                      + (" Its weight is tied to the Output Embedding." if m.share_embeddings else "")),
        "softmax": ("Softmax (outside the model)",
                    f"Transformer.forward returns logits {s_in}. Softmax over the last dim turns them into "
                    "next-token probabilities. During training this happens inside the cross-entropy loss."),
    }
    return table[key]
