"""
Run one real forward pass of the Vision Transformer on a small batch of synthetic images and
record a step-by-step execution trace for the UI (the generic tracing lives in shared/py/tracer.py).

Steps, in order:
    patch_embed   Conv2d patch projection + flatten + transpose   (glue lines in ImageEmbedding.forward)
    extra_linear  the Linear after the projection (not in the paper)
    cls_token     prepend [CLS]
    pe            + positional encoding
    per layer:    enc_norm1 → enc_self_attn → enc_res1 → enc_norm2 → enc_ffn → enc_res2
    enc_final     final LayerNorm
    cls_select    keep token 0 ([CLS])
    head          classification head (Linear D → K)
    softmax       class probabilities (outside the model)
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "shared" / "py"))
import tracer as tr  # noqa: E402
from tracer import DimNamer, Tracer, cube_slice, fmt, rl  # noqa: E402

from presets import downsample, make_images  # noqa: E402

# Project modules that read the configuration through `get_settings()`.
SETTINGS_MODULES = ["transformer", "encoder", "multiheadAttention", "feedForward",
                    "ImageEmbedding", "positionalEncoding"]

# trace key -> (diagram block key, component id, tensor role)
STEP_MAP = {
    "patch_embed": ("patch_embed", "patch_embedding", "src"),
    "extra_linear": ("extra_linear", "extra_linear", "src"),
    "cls_token": ("cls_token", "cls_token", "src"),
    "pe": ("pe", "positional_encoding", "src"),
    "enc_norm1": ("enc_norm1", "enc_norm", "src"),
    "enc_self_attn": ("enc_self_attn", "enc_self_attn", "src"),
    "enc_res1": ("enc_res1", "enc_residual", "src"),
    "enc_norm2": ("enc_norm2", "enc_norm", "src"),
    "enc_ffn": ("enc_ffn", "enc_ffn", "src"),
    "enc_res2": ("enc_res2", "enc_residual", "src"),
    "enc_final": ("encoder_stack", "encoder_stack", "src"),
    "cls_select": ("cls_select", "cls_select", "src"),
    "head": ("head", "head", "src"),
    "softmax": ("softmax", "softmax", "src"),
}

# Names for the operations a line performs (first matching pattern wins).
LABEL_RULES = [
    ("torch.matmul(q, k.transpose", "Dot products Q·Kᵀ, scaled by 1/√d_k"),
    ("torch.matmul(attn_weights", "Weighted sum of values: weights · V"),
    ("torch.softmax", "Softmax over the keys"),
    ("nan_to_num", "Zero any fully-masked rows"),
    ("x.view(batch_size, seq_len, self.n_heads", "Split into heads (view + transpose)"),
    (".contiguous().view", "Concatenate the heads (transpose + view)"),
    ("self.pe[:, :seq_len]", "Add the positional encoding"),
    ("patches.flatten(2)", "Flatten the patch grid: (B, D, H/P, W/P) → (B, D, N)"),
    ("patches.transpose(1, 2)", "Swap axes: (B, D, N) → (B, N, D)"),
    ("self.cls_token.expand", "Repeat [CLS] for every image (a view, no copy)"),
    ("torch.cat([cls_token, patches]", "Prepend [CLS]: (B, 1, D) + (B, N, D) → (B, N + 1, D)"),
    ("x + self.dropout1", "Residual add: x + MSA(LN(x))"),
    ("x + self.dropout2", "Residual add: x + MLP(LN(x))"),
    ("x[:, 0, :]", "Keep token 0, the [CLS] token"),
]
_label = tr.make_labeler(LABEL_RULES)


class ViTNamer(DimNamer):
    """Axis names, plus the ViT-only shapes: the conv output grid and 2-D (batch, ·) tensors."""

    def __call__(self, shape, label="", causal=False):
        if label.startswith("Conv2d"):
            return ["batch", "features", "patch rows", "patch cols"]
        if label.startswith("Flatten the patch grid"):
            return ["batch", "features", "patches"]
        if len(shape) == 2:
            if self.VOCAB.match(label):
                return ["batch", self.out_axis]
            if label.startswith("Keep token 0"):
                return ["batch", "features"]
        if len(shape) == 3 and label.startswith("Repeat [CLS]"):
            return ["batch", "seq", "features"]
        return super().__call__(shape, label, causal)


def simulate(settings, classes=None, images_kind="lines", detail: bool = False) -> dict:
    """Run one traced forward pass with `settings` (the project modules are pointed at them)."""
    with tr.detail_mode(detail), tr.use_settings(settings, SETTINGS_MODULES):
        return _simulate(settings, classes, images_kind)


def _simulate(settings, classes, images_kind) -> dict:
    from transformer import VisionTransformer  # imported lazily: needs sys.path set by caller

    m = settings.model
    torch.manual_seed(settings.training.seed)
    images, true = make_images(images_kind, m.image_size, m.in_channels)
    B, P, N = images.size(0), m.patch_size, m.n_patches
    names = classes or [f"class {k}" for k in range(m.num_classes)]

    model = VisionTransformer().eval()  # eval(): dropout disabled, so the trace is deterministic
    namer = ViTNamer(B, [N + 1, N], m.d_model, m.d_ff, m.n_heads, m.num_classes, out_axis="classes")
    layer_of = {id(layer): i for i, layer in enumerate(model.encoder.layers, start=1)}
    in_layer = lambda frame: layer_of.get(id(frame.f_locals.get("self")))  # noqa: E731
    glue = {}
    glue |= Tracer.glue_lines(ROOT, "ImageEmbedding.py", [
        {"match": "patches = self.patch_embedding(x)", "key": "patch_embed", "inputs": ["x"], "out": "patches"},
        {"match": "patches = patches.flatten(2)", "key": "patch_embed", "out": "patches"},
        {"match": "patches = patches.transpose(1, 2)", "key": "patch_embed", "out": "patches"},
        {"match": "patches = self.linear_embedding(patches)", "key": "extra_linear", "inputs": ["patches"], "out": "patches"},
        {"match": "cls_token = self.cls_token.expand", "key": "cls_token", "inputs": ["patches"], "out": "x"},
        {"match": "x = torch.cat([cls_token, patches]", "key": "cls_token", "out": "x"},
    ])
    glue |= Tracer.glue_lines(ROOT, "encoder.py", [
        {"match": "x = x + self.dropout1(attn_output)", "key": "enc_res1", "inputs": ["x", "attn_output"], "out": "x", "layer": in_layer},
        {"match": "x = x + self.dropout2(ffn_output)", "key": "enc_res2", "inputs": ["x", "ffn_output"], "out": "x", "layer": in_layer},
    ])
    glue |= Tracer.glue_lines(ROOT, "transformer.py", [
        {"match": "cls_output = x[:, 0, :]", "key": "cls_select", "inputs": ["x"], "out": "cls_output"},
    ])
    tracer = Tracer(ROOT, namer, _label, glue)

    def conv_sub(mod, inp, out):
        """Conv2d with kernel = stride = P is a matmul of the flattened patches with the reshaped kernel."""
        if not isinstance(mod, torch.nn.Conv2d):
            return None
        ops = None
        if tr.OPTS["detail"]:
            xp = F.unfold(inp[0], kernel_size=mod.kernel_size, stride=mod.stride).transpose(1, 2)  # (B, N, C·P·P)
            ops = {"x": rl(xp), "W": rl(mod.weight.reshape(mod.out_channels, -1)), "b": rl(mod.bias)}
        k = mod.kernel_size[0]
        return (f"Conv2d patch_embedding: {mod.in_channels}×{k}×{k} → {mod.out_channels}, stride {k}", "x_p · E + b", ops)

    tr.install_leaf_hooks(tracer, model, extra=conv_sub)

    def on_post(step, mod, args, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(mod, torch.nn.LayerNorm):
            tr.layernorm_subs(tracer, step, mod, args[0], out, post_norm=False)
        if isinstance(output, tuple) and len(output) > 1:
            # attention weights (B, H, T, S) -> average over heads per sample (first 16 tokens for ViT-Base)
            step["attn"] = output[1].mean(dim=1)[:, :16, :16].round(decimals=3).tolist()
        if mod is model.head:
            step["in2"] = rl(args[0][:, :64] if args[0].shape[-1] > 64 else args[0])
            step["out2"] = rl(out)

    for i, layer in enumerate(model.encoder.layers, start=1):
        tracer.hook(layer.norm1, "enc_norm1", i, on_post)
        tracer.hook(layer.self_attn, "enc_self_attn", i, on_post)
        tracer.hook(layer.norm2, "enc_norm2", i, on_post)
        tracer.hook(layer.ffn, "enc_ffn", i, on_post)
    tracer.hook(model.image_embedding.positional_encoding, "pe", None, on_post)
    tracer.hook(model.encoder.norm, "enc_final", None, on_post)
    tracer.hook(model.head, "head", None, on_post)

    try:
        sys.settrace(tracer.global_trace)
        with torch.no_grad():
            logits = model(images)
    finally:
        sys.settrace(None)

    # ---- the image itself, for the patch views ------------------------------------
    pe_step = next(s for s in tracer.steps if s["key"] == "patch_embed")
    thumbs = [downsample(img) for img in images]
    pe_step["extra"] = {
        "image": [rl(t) for t, _ in thumbs], "stride": thumbs[0][1],
        "size": m.image_size, "patch": P, "channels": m.in_channels, "grid": m.image_size // P,
    }

    # ---- softmax over the classes (applied outside the model) ----------------------
    probs = torch.softmax(logits, dim=-1)
    topk = []
    for b in range(B):
        p, idx = probs[b].topk(min(5, probs.size(-1)))
        topk.append({"pos": None, "ids": idx.tolist(), "probs": [round(v, 5) for v in p.tolist()]})
    head_step = next(s for s in tracer.steps if s["key"] == "head")
    lp, li = logits[0].topk(min(5, logits.size(-1)))
    head_step["topk_logits"] = {"pos": None, "ids": li.tolist(), "logits": [round(v, 3) for v in lp.tolist()]}
    sm_step = {"key": "softmax", "layer": None, "subs": []}
    tracer.stack.append(sm_step)
    mx = logits.max(dim=-1, keepdim=True).values
    ex = torch.exp(logits - mx)
    tot = ex.sum(dim=-1, keepdim=True)
    tracer.record_sub(mx, None, None, "Max logit per image (for numerical stability)", var="max", formula="m = max(z)")
    tracer.record_sub(ex, None, None, "Exponentiate the shifted logits", var="e", formula="e = exp(z − m)")
    tracer.record_sub(tot, None, None, "Sum over the classes", var="Σ", formula="Σ = sum(e)")
    tracer.record_sub(probs, None, None, "Divide: probabilities that sum to 1", var="p", formula="p = e / Σ")
    tracer.stack.pop()
    tracer.steps.append({
        "subs": sm_step["subs"], "key": "softmax", "layer": None, "frames": [], "exec": {},
        "inputs": [{"name": "logits", "shape": list(logits.shape), "dtype": "float32"}],
        "in_shapes": [list(logits.shape)], "out_shape": list(probs.shape), "out_dtype": "float32",
        "topk": topk, "in2": rl(logits), "out2": rl(probs),
    })
    for s in tracer.steps:                    # 2-D (batch, ·) values travel as in2 / out2
        if "in2" in s and "extra" not in s:
            s["extra"] = {"in2": s.pop("in2"), "out2": s.pop("out2")}

    nL = m.n_encoder_layers
    out = tr.finalize(tracer.steps, STEP_MAP, lambda key, L, ins, o: _describe(key, L, nL, ins, o, m),
                      lambda key: nL if key.startswith("enc") else None)

    # Token "ids" are just positions: 0 = [CLS], 1… = patches in raster order. Long sequences
    # (ViT-Base: 197 tokens) keep the first 16 positions, matching the cut tensors.
    T = N + 1 if N + 1 <= tr.OPTS["seq_cap"] or tr.OPTS["detail"] else tr.OPTS["seq_show"]
    g = m.image_size // P
    return {
        "batch": {
            "src": [list(range(T)) for _ in range(B)], "seq_total": N + 1,
            "src_vocab": {"0": "[CLS]", **{str(i): f"p{i}" for i in range(1, N + 1)}},
            "patch_rc": {str(i): [(i - 1) // g, (i - 1) % g] for i in range(1, N + 1)},
            "tgt_vocab": {str(k): n for k, n in enumerate(names)},
            "true": true, "pad": -1, "bos": 0, "eos": -1,
        },
        "steps": out,
    }


def _describe(key, L, nL, ins, o, m):
    s_in = fmt(ins[0]) if ins and ins[0] else ""
    s_out = fmt(o) if o else ""
    P, C, D, N = m.patch_size, m.in_channels, m.d_model, m.n_patches
    enc = f"Encoder layer {L}/{nL} · " if L else ""
    table = {
        "patch_embed": ("Patch embedding (linear projection of flattened patches)",
                        f"Cut the {m.image_size}×{m.image_size}×{C} image into N = {N} patches of {P}×{P}×{C} = {P * P * C} "
                        f"values and project each to D = {D}. A Conv2d with kernel = stride = {P} does exactly this; "
                        f"flatten + transpose then give {s_out}."),
        "extra_linear": ("Extra Linear (not in the paper)",
                         f"A Linear {D} → {D} on every patch embedding. The paper has no such layer: two linear maps "
                         "compose into one, so it adds parameters but no expressive power."),
        "cls_token": ("Prepend the [class] token",
                      f"A learned vector x_class is copied for every image and placed at position 0: {s_in} → {s_out}."),
        "pe": ("Positional encoding",
               f"Add one {D}-dim position vector per token (sinusoidal here; the paper learns them), then dropout "
               f"(off in eval mode). Shape stays {s_out}."),
        "enc_norm1": (enc + "LayerNorm before attention", "Pre-norm: LN is applied to the block's input; the residual uses the un-normalized x."),
        "enc_self_attn": (enc + "Multi-Head Self-Attention (MSA)",
                          f"Every token attends to every token (no masks): {m.n_heads} heads × {D // m.n_heads} dims."),
        "enc_res1": (enc + "Residual add (after MSA)", "z'ₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁ (Eq. 2)."),
        "enc_norm2": (enc + "LayerNorm before the MLP", "Pre-norm again, on z'ₗ."),
        "enc_ffn": (enc + "MLP", f"Linear {D} → {m.d_ff}, {m.activation.upper()}, Dropout, Linear {m.d_ff} → {D}, per token."),
        "enc_res2": (enc + "Residual add (after the MLP)", "zₗ = MLP(LN(z'ₗ)) + z'ₗ (Eq. 3)."),
        "enc_final": ("Final LayerNorm", f"y = LN(z_L), applied to every token {s_out}; only [CLS] is used next (Eq. 4)."),
        "cls_select": ("Keep the [CLS] token", f"x[:, 0, :] takes token 0 of every image: {s_in} → {s_out}."),
        "head": ("Classification head", f"One Linear layer D = {D} → K = {m.num_classes} classes: {s_in} → {s_out}."),
        "softmax": ("Softmax (outside the model)",
                    f"VisionTransformer.forward returns logits {s_in}. Softmax turns them into class probabilities; "
                    "during training this happens inside the cross-entropy loss."),
    }
    return table[key]
