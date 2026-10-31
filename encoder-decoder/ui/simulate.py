"""
Run one real forward pass of the Transformer on a small simulated batch and record a
step-by-step execution trace for the UI.

For each step we record:
  * which diagram block is running (and which encoder/decoder layer),
  * the exact project source lines that executed (captured with sys.settrace),
  * the call stack of project frames that led there,
  * input/output tensor shapes, a value preview, and (for attention) the weights.
"""

import contextlib
import importlib
import inspect
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PROJECT_FILES = {str(p.resolve()): p.name for p in ROOT.glob("*.py")}

# Default simulated translation batch (en → de). Sample 2 is shorter, so it gets padded.
SRC_SENTENCES = ["the cat sat on the mat", "I love you"]
TGT_SENTENCES = ["die Katze saß auf der Matte", "ich liebe dich"]

# Detail mode (tiny / small presets): keep every value, for every layer, plus matmul operands.
DETAIL = {"on": False}

# Project modules that read the configuration through `get_settings()`.
SETTINGS_MODULES = ["transformer", "encoder", "decoder", "multiheadAttention", "feedForward",
                    "inputEmbedding", "outputEmbedding", "positionalEncoding", "selfAttention"]


@contextlib.contextmanager
def use_settings(settings):
    """Temporarily make every project module see `settings` (no model code changes needed)."""
    saved = {}
    for name in SETTINGS_MODULES:
        mod = importlib.import_module(name)
        saved[name] = mod.get_settings
        mod.get_settings = lambda *a, **k: settings
    try:
        yield
    finally:
        for name, fn in saved.items():
            importlib.import_module(name).get_settings = fn


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


def _shape(x) -> list | None:
    return list(x.shape) if isinstance(x, torch.Tensor) else None


CUBE_DIMS = 16  # feature dims sampled per position for the tensor cube


def _r(v: float) -> float:
    """Round to 3 significant figures (keeps tiny probabilities like 3.1e-5). ±inf → ±1e30."""
    if v != v:
        return 0.0
    if v in (float("inf"), float("-inf")):
        return 1e30 if v > 0 else -1e30
    return float(f"{v:.3g}") if v else 0.0


def _rl(obj):
    return [_rl(o) for o in obj] if isinstance(obj, list) else _r(float(obj))


def _cube_slice(x) -> dict | None:
    """A small, evenly spaced slice of a (B, T, D) float tensor for the tensor cube.

    About every (D // CUBE_DIMS)-th feature is kept, so with D = 512 and 8 heads each head
    contributes exactly 2 of the 16 sampled dims.
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 3 or not x.is_floating_point():
        return None
    d = x.shape[-1]
    if DETAIL["on"]:
        return {"dims": list(range(d)), "D": d, "v": _rl(x.float().tolist())}
    k = min(CUBE_DIMS, d)
    # Alternate even/odd dims (0, 33, 64, 97, …) so positional-encoding sines (even) and
    # cosines (odd) both appear, while each head still contributes exactly 2 dims.
    idx = [min(d - 1, i * d // k + (i % 2)) for i in range(k)]
    return {"dims": idx, "D": d, "v": _rl(x[:, :, idx].float().tolist())}


class DimNamer:
    """Names tensor axes from their position and size (batch, heads, seq, features…)."""

    def __init__(self, B, src_len, tgt_len, d_model, d_ff, heads, vocab):
        self.B, self.lens = B, {src_len, tgt_len}
        self.sizes = {d_model: "features", d_ff: "hidden (d_ff)", d_model // heads: "head features",
                      vocab: "vocab"}

    def last(self, n: int, keys: bool) -> str:
        if keys and n in self.lens:
            return "keys"
        return self.sizes.get(n) or ("seq" if n in self.lens else "dim")

    KEYS_LAST = re.compile(r"^(Dot products|Add the mask|Softmax over|Zero any|Reshape padding|Merge causal|Boolean mask)")
    HEAD_LAST = re.compile(r"^(Split into heads|Weighted sum)")
    HIDDEN = re.compile(r"^(Linear net\[0\]|ReLU|GELU|SiLU)")
    VOCAB = re.compile(r"^(Max logit|Exponentiate|Sum over the vocabulary|Divide)")

    def __call__(self, shape: list[int], label: str = "", causal: bool = False) -> list[str]:
        n = len(shape)
        if n == 4:
            last = ("keys" if self.KEYS_LAST.match(label) else "head features" if self.HEAD_LAST.match(label)
                    else self.last(shape[3], keys=True))
            return ["batch", "heads", "queries", last]
        if n == 3:
            last = ("hidden (d_ff)" if self.HIDDEN.match(label) else "vocab" if self.VOCAB.match(label)
                    else "features" if label else self.last(shape[2], keys=False))
            return ["batch", "seq", last]
        if n == 2 and label:
            if label.startswith("Upper triangle") or causal:
                return ["queries", "keys"]
            if label.startswith(("Compare every token", "Boolean mask")):
                return ["batch", "keys" if label.startswith("Boolean") else "seq"]
        if n == 2:
            if shape[0] == self.B and shape[1] in self.lens:
                return ["batch", "seq"]
            if shape[0] == shape[1]:
                return ["queries", "keys"]
            return ["rows", "cols"]
        return ["dim"] * n


SUB_LAST = {3: 16, 4: 8}  # how many entries of the last axis to keep for 3-D / 4-D tensors


def _sub_slice(t: torch.Tensor, names: list[str]) -> dict:
    """A small, display-ready slice of any tensor (singleton axes squeezed, last axis sampled)."""
    shape = list(t.shape)
    keep = [i for i, n in enumerate(shape) if n != 1] or [len(shape) - 1]
    tt = t.detach().reshape([shape[i] for i in keep])
    nm, full = [names[i] for i in keep], [shape[i] for i in keep]
    idx = None
    cap = None if DETAIL["on"] else SUB_LAST.get(tt.dim())
    if cap and full[-1] > cap:
        idx = [i * full[-1] // cap for i in range(cap)]
        tt = tt.index_select(-1, torch.tensor(idx))
    kind = "bool" if t.dtype == torch.bool else "float" if t.is_floating_point() else "int"
    return {"v": _rl(tt.float().tolist()), "names": nm, "full": full, "idx": idx, "kind": kind}


def _short(code: str) -> str:
    code = code.strip()
    return code if len(code) <= 70 else code[:67] + "…"


def _label(code: str, var: str | None) -> str:
    """Human-readable name for the operation a line performs."""
    rules = [
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
    for pat, lab in rules:
        if pat in code:
            return lab
    return f"{var} = …" if var else _short(code)


def _fmt(shape) -> str:
    return "(" + ", ".join(map(str, shape)) + ")"


class Tracer:
    """Collects executed project lines and groups them into steps."""

    def __init__(self, namer=None):
        self.steps: list[dict] = []
        self.stack: list[dict] = []
        self.namer = namer
        self.pending: dict = {}   # frame -> (line being executed, {local name: id(tensor)})
        self.lines: dict[str, list[str]] = {}

    # ---- sub-steps ----------------------------------------------------------------
    def code_at(self, file: str, line: int) -> str:
        if file not in self.lines:
            self.lines[file] = (ROOT / file).read_text(encoding="utf-8").splitlines()
        return self.lines[file][line - 1]

    def record_sub(self, t, file: str | None, line: int | None, label: str, var: str | None = None,
                   formula: str | None = None, force: bool = False, ops: dict | None = None) -> None:
        """Add one tensor operation to the running step (deduplicated by tensor identity)."""
        if not self.stack or not isinstance(t, torch.Tensor):
            return
        step = self.stack[-1]
        # Keep strong references so a freed temporary's id can't be reused and mistaken for a seen tensor.
        seen = step.setdefault("_seen", {})
        if seen.get(id(t)) is t and not force:
            return
        seen[id(t)] = t
        sub = {"label": label, "var": var, "file": file, "line": line, "shape": list(t.shape),
               "dtype": str(t.dtype).replace("torch.", ""), "formula": formula}
        if file and line:
            sub["code"] = _short(self.code_at(file, line))
        subs = step.setdefault("subs", [])
        if (DETAIL["on"] or step["layer"] in (None, 1)) and self.namer:   # full detail, or layer 1 only
            # In masked self-attention the first boolean mask converted is the causal (T, T) one.
            causal = (label.startswith("Boolean mask") and step["key"] == "dec_masked_attn"
                      and not any(x["label"].startswith("Boolean mask") for x in subs))
            sub["vals"] = _sub_slice(t, self.namer(list(t.shape), label, causal))
            if ops and DETAIL["on"]:
                sub["ops"] = ops
        subs.append(sub)

    def _diff_locals(self, frame) -> None:
        prev = self.pending.get(frame)
        if prev is None or not self.stack:
            return
        line, snap = prev
        fname = PROJECT_FILES[frame.f_code.co_filename]
        code = self.code_at(fname, line)
        for name, val in frame.f_locals.items():
            if isinstance(val, torch.Tensor) and snap.get(name) != id(val):
                self.record_sub(val, fname, line, _label(code, name), var=name)

    # ---- sys.settrace plumbing -------------------------------------------------
    def global_trace(self, frame, event, arg):
        if event == "call" and frame.f_code.co_filename in PROJECT_FILES:
            return self.local_trace
        return None

    def local_trace(self, frame, event, arg):
        if event in ("line", "return"):
            self._diff_locals(frame)  # the previous line of this frame just finished
        if event == "line":
            if self.stack:
                fname = PROJECT_FILES[frame.f_code.co_filename]
                step = self.stack[-1]
                step["exec"].setdefault(fname, set()).add(frame.f_lineno)
                step["first"].setdefault(fname, frame.f_lineno)  # first line run, in time order
            self.pending[frame] = (frame.f_lineno, {k: id(v) for k, v in frame.f_locals.items()
                                                     if isinstance(v, torch.Tensor)})
        elif event == "return":
            prev = self.pending.pop(frame, None)
            if prev and self.stack and isinstance(arg, torch.Tensor):
                fname = PROJECT_FILES[frame.f_code.co_filename]
                code = self.code_at(fname, prev[0])
                self.record_sub(arg, fname, prev[0], _label(code, None), var="return")
        return self.local_trace

    def current_line(self) -> tuple[str | None, int | None]:
        """Innermost project frame's current line (the code that triggered a PyTorch op)."""
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename in PROJECT_FILES:
                return PROJECT_FILES[f.f_code.co_filename], f.f_lineno
            f = f.f_back
        return None, None

    # ---- step boundaries -------------------------------------------------------
    @staticmethod
    def _project_frames() -> list[dict]:
        frames = []
        f = sys._getframe(2)
        while f is not None:
            if f.f_code.co_filename in PROJECT_FILES:
                frames.append({
                    "file": PROJECT_FILES[f.f_code.co_filename],
                    "line": f.f_lineno,
                    "func": getattr(f.f_code, "co_qualname", f.f_code.co_name),
                })
            f = f.f_back
        return frames[::-1]  # outermost first

    def begin(self, key: str, layer: int | None, inputs: list[tuple[str, object]], module=None) -> dict:
        step = {
            "key": key,
            "layer": layer,
            "_module": module,
            "frames": self._project_frames(),
            "exec": {},
            "first": {},
            # Labelled tensor inputs, e.g. [{"name": "query", "shape": [2, 7, 512], "dtype": "float32"}]
            "inputs": [
                {"name": n, "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}
                for n, x in inputs if isinstance(x, torch.Tensor)
            ],
        }
        step["in_shapes"] = [i["shape"] for i in step["inputs"]]
        # Inputs aren't operations: aliases of them (e.g. `key = query`) are skipped as sub-steps.
        step["_seen"] = {id(x): x for _, x in inputs if isinstance(x, torch.Tensor)}
        self.stack.append(step)
        return step

    def end(self, output) -> dict:
        step = self.stack.pop()
        self.steps.append(step)
        return step


def _layernorm_subs(tracer: Tracer, step: dict, mod, x: torch.Tensor, out: torch.Tensor) -> None:
    """Decompose LayerNorm (computed in C++ by PyTorch) into its arithmetic, linked to the call site."""
    call = step["frames"][-1] if step["frames"] else {"file": None, "line": None}
    file, line = call["file"], call["line"]
    tracer.stack.append(step)  # record into this (already finished) step
    try:
        if not step["key"].endswith("final"):
            tracer.record_sub(x, file, line, "Residual add: x + Dropout(Sublayer(x))", var="x",
                              formula="x + Sublayer(x)", force=True)
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        xhat = (x - mu) / torch.sqrt(var + mod.eps)
        tracer.record_sub(mu, file, line, "Mean over the features", var="μ", formula="μ = mean(x)")
        tracer.record_sub(var, file, line, "Variance over the features", var="σ²", formula="σ² = mean((x − μ)²)")
        tracer.record_sub(xhat, file, line, "Normalize", var="x̂", formula="x̂ = (x − μ) / √(σ² + ε)")
        tracer.record_sub(out, file, line, "Scale and shift (learned γ, β)", var="y", formula="y = γ · x̂ + β",
                          ops={"gamma": _rl(mod.weight.detach().tolist()), "beta": _rl(mod.bias.detach().tolist()),
                               "eps": mod.eps})
    finally:
        tracer.stack.pop()


def simulate(settings, src_sentences=None, tgt_sentences=None, detail: bool = False, src_bos: bool = True) -> dict:
    """Run one traced forward pass with `settings` (the project modules are pointed at them)."""
    DETAIL["on"] = detail
    try:
        with use_settings(settings):
            return _simulate(settings, src_sentences or SRC_SENTENCES, tgt_sentences or TGT_SENTENCES, src_bos)
    finally:
        DETAIL["on"] = False


def _definition_line(model, name: str) -> dict | None:
    """Where a PyTorch layer is defined in the project (`self.q_proj = nn.Linear(...)`, or the
    k-th entry of an `nn.Sequential(...)`), so the code panel can show it next to the call site."""
    parts = name.split(".")
    owner_path, attr, seq_index = parts[:-1], parts[-1], None
    if attr.isdigit():                                  # an element of an nn.Sequential
        seq_index, attr = int(attr), owner_path[-1]
        owner_path = owner_path[:-1]
    owner = model.get_submodule(".".join(owner_path)) if owner_path else model
    try:
        src, start = inspect.getsourcelines(type(owner))
        file = PROJECT_FILES.get(str(Path(inspect.getsourcefile(type(owner))).resolve()))
    except (OSError, TypeError):
        return None
    if not file:
        return None
    for i, line in enumerate(src):
        if re.search(rf"self\.{re.escape(attr)}\s*=", line):
            if seq_index is None:
                return {"file": file, "line": start + i}
            # count the elements inside nn.Sequential( ... )
            k = -1
            for j in range(i + 1, len(src)):
                text = src[j].strip()
                if not text or text.startswith(")"):
                    break
                k += 1
                if k == seq_index:
                    return {"file": file, "line": start + j}
    return None


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
    namer = DimNamer(src.size(0), src.size(1), tgt.size(1), m.d_model, m.d_ff, m.n_heads, m.tgt_vocab_size)
    tracer = Tracer(namer)

    # Leaf PyTorch layers (Linear, Embedding, activations) become sub-steps of the running step.
    names = {mod: n for n, mod in model.named_modules()}

    def leaf_hook(mod, inp, out):
        if not tracer.stack or mod is tracer.stack[-1].get("_module"):
            return
        if isinstance(mod, (torch.nn.Dropout, torch.nn.LayerNorm)):
            return  # dropout is the identity in eval mode; LayerNorm steps are decomposed below
        file, line = tracer.current_line()
        attr = names[mod].split(".")[-1]
        attr = f"net[{attr}]" if attr.isdigit() else attr
        ops = None
        if isinstance(mod, torch.nn.Linear):
            label, formula = f"Linear {attr}: {mod.in_features} → {mod.out_features}", "x · Wᵀ + b"
            if DETAIL["on"]:   # operands for the cell-by-cell matmul view
                ops = {"x": _rl(inp[0].detach().float().tolist()), "W": _rl(mod.weight.detach().tolist()),
                       "b": _rl(mod.bias.detach().tolist()) if mod.bias is not None else None}
        elif isinstance(mod, torch.nn.Embedding):
            label, formula = "Embedding lookup: one table row per token id", "E[token]"
            if DETAIL["on"]:
                used = sorted({int(i) for i in inp[0].flatten().tolist()})
                ops = {"rows": {str(i): _rl(mod.weight[i].detach().tolist()) for i in used}}
        else:
            label, formula = f"{type(mod).__name__} activation", f"{type(mod).__name__.lower()}(x)"
            if DETAIL["on"]:
                ops = {"x": _rl(inp[0].detach().float().tolist())}
        tracer.record_sub(out, file, line, label, var=attr, formula=formula, ops=ops)
        if tracer.stack and tracer.stack[-1].get("subs") and tracer.stack[-1]["subs"][-1]["label"] == label:
            d = _definition_line(model, names[mod])
            if d:
                tracer.stack[-1]["subs"][-1]["def"] = d

    for mod in model.modules():
        if not list(mod.children()):
            mod.register_forward_hook(leaf_hook)
    pe_calls = {"n": 0}
    extras: dict[int, dict] = {}  # id(step) -> extra payload

    # ---- forward hooks on the modules that correspond to diagram blocks ---------
    def hook(module, key, layer=None):
        def pre(mod, args, kwargs):
            k = key
            if key == "pe":
                k = "pe_enc" if pe_calls["n"] == 0 else "pe_dec"
                pe_calls["n"] += 1
            names = [p for p in inspect.signature(mod.forward).parameters]
            named = list(zip(names, args)) + list(kwargs.items())
            step = tracer.begin(k, layer, named, module=mod)
            first = next((t for _, t in named if isinstance(t, torch.Tensor)), None)
            step["in_vals"] = _cube_slice(first)

        def post(mod, args, kwargs, output):
            step = tracer.end(output)
            out = output[0] if isinstance(output, tuple) else output
            step["out_shape"] = _shape(out)
            step["out_dtype"] = str(out.dtype).replace("torch.", "")
            step["out_vals"] = _cube_slice(out)
            if DETAIL["on"] and isinstance(mod, torch.nn.Linear):   # e.g. the generator: operands for the matmul view
                step["ops"] = {"x": _rl(args[0].detach().float().tolist()), "W": _rl(mod.weight.detach().tolist()),
                               "b": _rl(mod.bias.detach().tolist()) if mod.bias is not None else None}
            if isinstance(mod, torch.nn.LayerNorm):
                _layernorm_subs(tracer, step, mod, args[0], out)
            if isinstance(output, tuple) and len(output) > 1:
                # attention weights (B, H, T, S) -> average over heads per sample
                step["attn"] = output[1].mean(dim=1).round(decimals=3).tolist()

        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.register_forward_hook(post, with_kwargs=True)

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
        "topk": topk, "in_vals": _cube_slice(logits), "out_vals": _cube_slice(probs),
    })

    # ---- finalize: titles, descriptions, primary file ---------------------------
    d, h, V = m.d_model, m.n_heads, m.tgt_vocab_size
    nE, nD = m.n_encoder_layers, m.n_decoder_layers
    out = []
    for s in tracer.steps:
        block, comp, role = STEP_MAP[s["key"]]
        ins = s.get("in_shapes") or [[]]
        o = s.get("out_shape") or []
        L = s["layer"]
        title, desc = _describe(s["key"], L, nE, nD, ins, o, d, h, m, tok, V)
        exec_ = {f: sorted(v) for f, v in s["exec"].items()}
        if exec_:
            primary = max(exec_, key=lambda f: len(exec_[f]))
        elif s["frames"]:
            primary = s["frames"][-1]["file"]
            exec_ = {primary: [s["frames"][-1]["line"]]}  # e.g. LayerNorm: show the call site
        else:
            primary = None
        step = {
            "key": s["key"], "block": block, "comp": comp, "role": role,
            "layer": L, "nlayers": nE if s["key"].startswith("enc") else nD if s["key"].startswith("dec") else None,
            "title": title, "desc": desc, "frames": s["frames"], "exec": exec_, "primary": primary,
            "first": s.get("first", {}),
            "inputs": s.get("inputs", []) + s.get("scalar_inputs", []),
            "out_shape": o, "out_dtype": s.get("out_dtype"),
        }
        for k in ("attn", "mask", "topk", "topk_logits", "in_vals", "out_vals", "subs", "ops"):
            if k in s:
                step[k] = s[k]
        out.append(step)

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
