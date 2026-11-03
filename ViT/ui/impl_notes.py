"""
Implementation details in this codebase that the paper doesn't describe.

Each note is pinned to code by text patterns (not line numbers), so it keeps working when
lines move. `ranges` is a list of (start_pattern, end_pattern | None): the range runs from
the first line containing start_pattern to the next line containing end_pattern.
`comps` lists the diagram components / step keys the note belongs to.
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTES = [
    # ---------------- positional encoding ----------------
    {
        "id": "pe_log", "file": "positionalEncoding.py", "comps": ["positional_encoding"],
        "ranges": [("# Compute 1/10000^(2i/d)", ")  # shape: (d_model/2,)")],
        "title": "Frequencies computed in log space (exp of a log)",
        "text": "The original Transformer paper writes the frequency as 1 / 10000^(2i/d_model). The code rewrites it by "
                "taking the ln, simplifying the power into a product, and applying exp to get the same "
                "value back (derivation below). The result is one arange, one multiply and one exp over "
                "the whole vector. Using exp/log instead of raising a large base to a power is the usual "
                "numerically safe idiom. Here 10000^(2i/d) never exceeds 10000, so float32 overflow isn't "
                "a real risk; it would matter with a much larger base.",
    },
    {
        "id": "pe_vec", "file": "positionalEncoding.py", "comps": ["positional_encoding"],
        "ranges": [("position = torch.arange", None), ("pe[:, 0::2]", "pe[:, 1::2]")],
        "title": "Vectorized: every position × every frequency at once",
        "text": "position has shape (max_len, 1) and div_term has shape (d_model/2,). Broadcasting "
                "position * div_term builds the whole (max_len, d_model/2) table of angles in one "
                "operation, instead of two nested Python loops over pos and i. Slicing with 0::2 and "
                "1::2 then writes the sines into the even columns and the cosines into the odd columns, "
                "matching the 2i / 2i+1 indices of the original formula.",
    },
    {
        "id": "pe_buffer", "file": "positionalEncoding.py", "comps": ["positional_encoding"],
        "ranges": [("# persistent=False", 'self.register_buffer("pe"')],
        "title": "Precomputed once and stored as a buffer",
        "text": "The table is built a single time, at construction, for max_seq_len positions. "
                "register_buffer makes it move with model.to(device) without being a trainable "
                "parameter, and persistent=False keeps it out of checkpoints, since it can always be "
                "recomputed.",
    },
    {
        "id": "pe_slice", "file": "positionalEncoding.py", "comps": ["positional_encoding"],
        "ranges": [("x = x + self.pe[:, :seq_len]", None)],
        "title": "Slice and broadcast over the batch",
        "text": "pe is stored as (1, max_len, d_model). The forward pass slices the first seq_len rows "
                "to get (1, T, d_model), which broadcasts across the batch dimension when added to x "
                "(B, T, d_model). The .to(dtype) keeps fp16/bf16 inputs in their own precision.",
    },
    {
        "id": "pe_sinusoidal", "file": "positionalEncoding.py", "comps": ["positional_encoding"],
        "ranges": [('if self.encoding_type == "sinusoidal":', 'self.register_buffer("pe"')],
        "title": "Fixed sinusoids instead of learned position embeddings",
        "text": "ViT learns E_pos ∈ ℝ^((N+1)×D) as a parameter (Eq. 1). config.yml selects the sinusoidal table of "
                "the original Transformer instead, so nothing here is trained. The 'learned' branch below is the "
                "paper's version; Appendix D.4 found the choice of encoding makes little difference.",
    },
    # ---------------- patch embedding ----------------
    {
        "id": "conv_patch", "file": "ImageEmbedding.py", "comps": ["patch_embedding", "image_embedding"],
        "ranges": [("self.patch_embedding = nn.Conv2d(", "stride=cfg.patch_size,")],
        "title": "The patch projection E as a Conv2d",
        "text": "With kernel = stride = P, the convolution visits each non-overlapping P×P patch once and takes a "
                "dot product of its P²·C pixels with each of the D kernels. That is x_p·E + b for every patch, "
                "with E = weight.reshape(D, P²·C)ᵀ, computed without materializing the flattened patches. It is not "
                "a CNN backbone (the paper's hybrid model): there is one linear map, no non-linearity, no overlap.",
    },
    {
        "id": "flatten_transpose", "file": "ImageEmbedding.py", "comps": ["patch_embedding"],
        "ranges": [("patches = patches.flatten(2)", None), ("patches = patches.transpose(1, 2)", None)],
        "title": "From a grid of patches to a sequence",
        "text": "The convolution returns a (B, D, H/P, W/P) grid. flatten(2) joins the two grid axes in raster "
                "order (row by row), the same order the paper's 1-D position embeddings assume, and transpose "
                "puts the sequence axis before the features: (B, N, D).",
    },
    {
        "id": "extra_linear", "file": "ImageEmbedding.py", "comps": ["extra_linear"],
        "ranges": [("# Extra projection of the patch embeddings", "self.linear_embedding = nn.Linear")],
        "title": "Extra Linear after the projection (not in the paper)",
        "text": "Eq. 1 has a single projection E. Linear(D → D) after it gives x_p·E·Wᵀ + (b·Wᵀ + b′), which is "
                "again one affine map of the patch, so the model can represent nothing new; it only adds D² + D "
                "parameters. It is kept on purpose in this repository and flagged on the diagram.",
    },
    {
        "id": "cls_init", "file": "ImageEmbedding.py", "comps": ["cls_token"],
        "ranges": [("self.cls_token = nn.Parameter(", "nn.init.normal_(self.cls_token")],
        "title": "[CLS] initialized from N(0, 0.02²)",
        "text": "The paper doesn't state an initialization for x_class. A small normal init is the common "
                "choice; _reset_parameters skips it, so the Xavier init doesn't overwrite it.",
    },
    {
        "id": "cls_expand", "file": "ImageEmbedding.py", "comps": ["cls_token"],
        "ranges": [("cls_token = self.cls_token.expand(", None)],
        "title": "expand: one [CLS] for every image, without copying",
        "text": "expand returns a (B, 1, D) view whose batch stride is 0: every image reads the same D numbers, so "
                "gradients from the whole batch accumulate into the single parameter. torch.cat then makes the "
                "real (B, N + 1, D) tensor.",
    },
    # ---------------- encoder ----------------
    {
        "id": "pre_norm", "file": "encoder.py", "comps": ["enc_norm", "enc_residual", "encoder_layer"],
        "ranges": [("attn_output, _ = self.self_attn(self.norm1(x))", "x = x + self.dropout2(ffn_output)")],
        "title": "Pre-norm residuals, with dropout on the branch",
        "text": "The normalized copy only feeds the sub-layer; the residual adds the un-normalized x, so the "
                "identity path runs untouched from the patches to the final LayerNorm. The dropout on each "
                "branch (off in eval mode) follows common practice; the paper only mentions dropout in its training setup.",
    },
    {
        "id": "mha_split", "file": "multiheadAttention.py", "comps": ["enc_self_attn"],
        "ranges": [("def _split_heads", "return x.view(")],
        "title": "All heads in one batched matmul",
        "text": "Appendix A describes k attention heads running in parallel. The code reshapes (B, N+1, D) → "
                "(B, k, N+1, D/k) with view + transpose, so a single torch.matmul computes every head at once. "
                "view doesn't copy data.",
    },
    {
        "id": "mha_nomask", "file": "multiheadAttention.py", "comps": ["enc_self_attn"],
        "ranges": [("mask = merge_masks(", "scores = scores + mask")],
        "title": "No masks for images",
        "text": "Every image in the batch has the same N + 1 tokens, so there is no padding and no causal order: "
                "merge_masks returns None and the scores are used as they are. The mask code is a leftover of the "
                "text Transformer this module was written for, and costs nothing here.",
    },
    {
        "id": "mha_concat", "file": "multiheadAttention.py", "comps": ["enc_self_attn"],
        "ranges": [("context = context.transpose(1, 2).contiguous()", None)],
        "title": "Concat(head₁ … head_k) as a reshape",
        "text": "Concatenating the heads is done by transposing back to (B, N+1, k, D/k) and viewing as "
                "(B, N+1, D). contiguous() is needed because transpose leaves the memory in a strided layout "
                "that view can't reinterpret.",
    },
    {
        "id": "ffn_dropout", "file": "feedForward.py", "comps": ["enc_ffn"],
        "ranges": [("nn.Dropout(cfg.dropout),", None)],
        "title": "Dropout inside the MLP",
        "text": "The paper only says the MLP has two layers with a GELU. The dropout between them is common "
                "practice (it is off in eval mode, so the trace is unaffected).",
    },
    {
        "id": "gelu_exact", "file": "feedForward.py", "comps": ["enc_ffn"],
        "ranges": [('"gelu": nn.GELU,', None)],
        "title": "Exact GELU, x·Φ(x)",
        "text": "nn.GELU() defaults to the exact form with the error function, x·½(1 + erf(x/√2)); the tanh "
                "approximation is available with approximate='tanh'. The page recomputes the exact form.",
    },
    # ---------------- model ----------------
    {
        "id": "xavier", "file": "transformer.py", "comps": ["vit", "head"],
        "ranges": [("def _reset_parameters", "nn.init.xavier_uniform_")],
        "title": "Xavier-uniform initialization (except [CLS])",
        "text": "Xavier-uniform for every weight matrix and the conv kernel. The paper fine-tunes with a "
                "zero-initialized head; this code starts from scratch, so the head is Xavier-initialized too.",
    },
    {
        "id": "cls_after_norm", "file": "transformer.py", "comps": ["cls_select", "encoder_stack"],
        "ranges": [("cls_output = x[:, 0, :]", None)],
        "title": "[CLS] is taken after the final LayerNorm",
        "text": "Encoder.forward ends with LayerNorm over every token, so x[:, 0] is already y = LN(z_L⁰) of Eq. 4. "
                "Normalizing all N + 1 tokens costs a little extra but keeps the encoder usable for other heads.",
    },
]


def _pe_log_math(d: int) -> str:
    """Step-by-step derivation of the log-space div_term, with a numeric check for this d_model."""
    ln10k = math.log(10000.0)
    i = 1
    paper = 10000.0 ** (-2 * i / d)
    code = math.exp(-(2 * i / d) * ln10k)
    c = lambda t: f'<code class="mono">{t}</code>'  # noqa: E731
    return f"""
<ol class="deriv">
  <li><span class="lbl">Original Transformer (§3.5)</span> the frequency of dimension pair i is
      <span class="m">div<sub>i</sub> = 1 / 10000<sup>2i/d</sup> = 10000<sup>−2i/d</sup></span></li>
  <li><span class="lbl">Take ln of both sides</span>
      <span class="m">ln(div<sub>i</sub>) = ln(10000<sup>−2i/d</sup>)</span></li>
  <li><span class="lbl">Simplify with ln(a<sup>b</sup>) = b · ln a</span>
      <span class="m">ln(div<sub>i</sub>) = −(2i/d) · ln 10000</span>
      <span class="why">The power is gone: it's now a product, and ln 10000 ≈ {ln10k:.4f} is just a constant.</span></li>
  <li><span class="lbl">Apply exp to undo the ln</span> (e<sup>ln x</sup> = x)
      <span class="m">div<sub>i</sub> = exp(−(2i/d) · ln 10000)</span></li>
</ol>
<table class="dmap">
  <tr><th>code</th><th>math</th></tr>
  <tr><td>{c("torch.arange(0, self.d_model, 2)")}</td><td>2i = 0, 2, 4, …, {d - 2} (one entry per sin/cos pair)</td></tr>
  <tr><td>{c("-… / self.d_model")}</td><td>−2i / d</td></tr>
  <tr><td>{c("* torch.log(torch.tensor(10000.0))")}</td><td>· ln 10000</td></tr>
  <tr><td>{c("torch.exp(…)")}</td><td>exp( … ), i.e. step 4</td></tr>
</table>
<p class="check">Check with d = {d}, i = {i}: 10000<sup>−2/{d}</sup> = {paper:.10f} and
exp(−(2/{d}) · {ln10k:.4f}) = {code:.10f}: the same value.</p>"""


def resolve_notes(d_model: int = 512) -> list[dict]:
    """Turn each note's text patterns into concrete line numbers. Fails loudly if stale."""
    math_html = {"pe_log": _pe_log_math(d_model)}
    out = []
    for n in NOTES:
        lines = (ROOT / n["file"]).read_text(encoding="utf-8").splitlines()
        found, pos = [], 0
        for start, end in n["ranges"]:
            s = next((i for i in range(pos, len(lines)) if start in lines[i]), None)
            if s is None:
                raise SystemExit(f"[build_ui] Implementation note '{n['id']}': '{start}' not found in {n['file']}")
            e = s if end is None else next((i for i in range(s, len(lines)) if end in lines[i]), None)
            if e is None:
                raise SystemExit(f"[build_ui] Implementation note '{n['id']}': '{end}' not found in {n['file']}")
            found.extend(range(s + 1, e + 2))  # 1-based, inclusive
            pos = e + 1
        out.append({k: n[k] for k in ("id", "file", "comps", "title", "text")}
                   | {"lines": found, "math": math_html.get(n["id"])})
    return out
