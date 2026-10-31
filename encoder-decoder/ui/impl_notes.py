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
        "text": "The paper writes the frequency as 1 / 10000^(2i/d_model). The code rewrites it by "
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
                "matching the 2i / 2i+1 indices in the paper's formula.",
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
    # ---------------- attention ----------------
    {
        "id": "mha_split", "file": "multiheadAttention.py",
        "comps": ["enc_self_attn", "dec_masked_attn", "dec_cross_attn"],
        "ranges": [("def _split_heads", "return x.view(")],
        "title": "All heads in one batched matmul",
        "text": "The paper describes h separate attention functions running in parallel. The code "
                "reshapes (B, T, d_model) → (B, h, T, d_k) with view + transpose, so a single "
                "torch.matmul computes every head at once. view doesn't copy data.",
    },
    {
        "id": "mha_masks", "file": "multiheadAttention.py",
        "comps": ["enc_self_attn", "dec_masked_attn", "dec_cross_attn", "padding_mask", "causal_mask"],
        "ranges": [("kpm = _to_additive", None), ("mask = merge_masks(", "scores = scores + mask")],
        "title": "Masks become one additive −∞ tensor",
        "text": "The paper only says illegal connections are set to −∞. Here boolean masks (True = "
                "blocked) are converted to 0 / −∞ and added to the scores. The padding mask is "
                "reshaped to (B, 1, 1, S) so it broadcasts over every head and every query, and the "
                "causal and padding masks are merged into one tensor before the softmax.",
    },
    {
        "id": "mha_nan", "file": "multiheadAttention.py",
        "comps": ["enc_self_attn", "dec_masked_attn", "dec_cross_attn"],
        "ranges": [("# Rows where every key is masked", "torch.nan_to_num")],
        "title": "Fully masked rows",
        "text": "If every key in a row is masked, the softmax of all −∞ is 0/0 = NaN. nan_to_num "
                "turns those rows into zeros so one padded query can't poison the whole batch.",
    },
    {
        "id": "mha_concat", "file": "multiheadAttention.py",
        "comps": ["enc_self_attn", "dec_masked_attn", "dec_cross_attn"],
        "ranges": [("context = context.transpose(1, 2).contiguous()", None)],
        "title": "Concat(head₁ … head_h) as a reshape",
        "text": "Concatenating the heads is done by transposing back to (B, T, h, d_k) and viewing as "
                "(B, T, d_model). contiguous() is needed because transpose leaves the memory in a "
                "strided layout that view can't reinterpret.",
    },
    # ---------------- masks ----------------
    {
        "id": "causal_triu", "file": "transformer.py", "comps": ["causal_mask", "dec_masked_attn"],
        "ranges": [("return torch.triu(", "diagonal=1")],
        "title": "Causal mask with triu(diagonal=1)",
        "text": "One call builds the whole T × T mask: True strictly above the diagonal, i.e. every "
                "future position. No loops, and it's created on the same device as the targets.",
    },
    {
        "id": "pad_eq", "file": "transformer.py", "comps": ["padding_mask"],
        "ranges": [("return tokens == pad_token_id", None)],
        "title": "Padding mask by elementwise comparison",
        "text": "tokens == pad_token_id gives a (B, T) boolean tensor directly. The paper doesn't "
                "discuss padding at all.",
    },
    # ---------------- model setup ----------------
    {
        "id": "xavier", "file": "transformer.py", "comps": ["transformer", "linear"],
        "ranges": [("def _reset_parameters", "nn.init.xavier_uniform_")],
        "title": "Xavier-uniform initialization",
        "text": "The paper doesn't specify an initialization. This code uses Xavier-uniform for every "
                "weight matrix, as torch.nn.Transformer does.",
    },
    {
        "id": "tie_after_init", "file": "transformer.py", "comps": ["linear", "output_embedding", "transformer"],
        "ranges": [("self._reset_parameters()", "self.generator.weight = self.output_embedding.weight")],
        "title": "Weights tied after initialization",
        "text": "The generator's weight is replaced by the output embedding's tensor after init, so both "
                "names point at the same Parameter, and model.parameters() counts it once.",
    },
    {
        "id": "ffn_dropout", "file": "feedForward.py", "comps": ["enc_ffn", "dec_ffn"],
        "ranges": [("nn.Dropout(cfg.dropout),", None)],
        "title": "Dropout inside the feed-forward network",
        "text": "Eq. 2 has no dropout between the two linear layers; §5.4 only applies dropout to "
                "sub-layer outputs. This extra dropout follows common implementations.",
    },
    {
        "id": "enc_final_norm", "file": "encoder.py", "comps": ["encoder_stack", "encoder_layer"],
        "ranges": [("self.norm = nn.LayerNorm", None)],
        "title": "Extra LayerNorm after the encoder stack",
        "text": "Not in the paper's post-norm design. It's what torch.nn.Transformer does, and it "
                "normalizes the memory that the decoder attends to.",
    },
    {
        "id": "dec_final_norm", "file": "decoder.py", "comps": ["decoder_stack", "decoder_layer"],
        "ranges": [("self.norm = nn.LayerNorm", None)],
        "title": "Extra LayerNorm after the decoder stack",
        "text": "Not in the paper. It normalizes the decoder output before the final Linear layer.",
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
  <li><span class="lbl">Paper (§3.5)</span> the frequency of dimension pair i is
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
