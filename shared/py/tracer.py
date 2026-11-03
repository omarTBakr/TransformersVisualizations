"""
Shared tracing core for the visualizations: run one real forward pass and record a
step-by-step execution trace.

For each step we record:
  * which diagram block is running (and which layer),
  * the exact project source lines that executed (captured with sys.settrace),
  * the call stack of project frames that led there,
  * input/output tensor shapes, a value preview, and sub-steps (one per tensor operation),
    with the operands of every matmul in detail mode so the page can check the arithmetic.

A model's ui/simulate.py builds the batch, hooks the modules that correspond to its diagram
blocks, and turns the raw trace into titled steps. Everything model-independent lives here.

Steps come from three places:
  * forward hooks on modules (`Tracer.begin` / `Tracer.end`),
  * "glue" lines: tensor code outside any hooked module, e.g. a residual `x = x + …` or
    `x[:, 0]`; the model names the lines and they become steps of their own,
  * steps the model builds by hand (e.g. softmax, which runs outside the model).
"""

import contextlib
import importlib
import inspect
import re
import sys
from pathlib import Path

import torch

# Detail mode (tiny / small presets): keep every value, for every layer, plus matmul operands.
# Otherwise feature axes are sampled, and long sequences are cut to their first positions.
OPTS = {"detail": False, "seq_cap": 32, "seq_show": 16}
CUBE_DIMS = 16              # feature dims sampled per position for the tensor cube
SUB_LAST = {3: 16, 4: 8}    # how many entries of the last axis to keep for 3-D / 4-D sub-step tensors
SEQ_AXES = {"seq", "queries", "keys"}


@contextlib.contextmanager
def use_settings(settings, modules: list[str]):
    """Temporarily make every listed project module see `settings` (no model code changes needed)."""
    saved = {}
    for name in modules:
        mod = importlib.import_module(name)
        saved[name] = mod.get_settings
        mod.get_settings = lambda *a, **k: settings
    try:
        yield
    finally:
        for name, fn in saved.items():
            importlib.import_module(name).get_settings = fn


@contextlib.contextmanager
def detail_mode(on: bool):
    OPTS["detail"] = on
    try:
        yield
    finally:
        OPTS["detail"] = False


def shape(x) -> list | None:
    return list(x.shape) if isinstance(x, torch.Tensor) else None


def r3(v: float) -> float:
    """Round to 3 significant figures (keeps tiny probabilities like 3.1e-5). ±inf → ±1e30."""
    if v != v:
        return 0.0
    if v in (float("inf"), float("-inf")):
        return 1e30 if v > 0 else -1e30
    return float(f"{v:.3g}") if v else 0.0


def rl(obj):
    """r3 applied to a nested list (or a tensor)."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().float().tolist()
    return [rl(o) for o in obj] if isinstance(obj, list) else r3(float(obj))


def _seq_cut(n: int) -> int | None:
    """Positions to keep along a sequence axis of length n (None = keep all)."""
    if OPTS["detail"] or n <= OPTS["seq_cap"]:
        return None
    return OPTS["seq_show"]


def cube_slice(x) -> dict | None:
    """A small, evenly spaced slice of a (B, T, D) float tensor for the tensor cube.

    About every (D // CUBE_DIMS)-th feature is kept, so with D = 512 and 8 heads each head
    contributes exactly 2 of the 16 sampled dims. Sequences longer than OPTS["seq_cap"] keep
    their first OPTS["seq_show"] positions ("T" then holds the full length).
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 3 or not x.is_floating_point():
        return None
    d = x.shape[-1]
    if OPTS["detail"]:
        return {"dims": list(range(d)), "D": d, "v": rl(x)}
    cut, full_t = _seq_cut(x.shape[1]), x.shape[1]
    if cut:
        x = x[:, :cut]
    k = min(CUBE_DIMS, d)
    # Alternate even/odd dims (0, 33, 64, 97, …) so positional-encoding sines (even) and
    # cosines (odd) both appear, while each head still contributes exactly 2 dims.
    idx = [min(d - 1, i * d // k + (i % 2)) for i in range(k)]
    out = {"dims": idx, "D": d, "v": rl(x[:, :, idx])}
    if cut:
        out["T"] = full_t
    return out


class DimNamer:
    """Names tensor axes from their position and size (batch, heads, seq, features…)."""

    KEYS_LAST = re.compile(r"^(Dot products|Add the mask|Softmax over|Zero any|Reshape padding|Merge causal|Boolean mask)")
    HEAD_LAST = re.compile(r"^(Split into heads|Weighted sum)")
    HIDDEN = re.compile(r"^(Linear net\[0\]|ReLU|GELU|SiLU)")
    VOCAB = re.compile(r"^(Max logit|Exponentiate|Sum over the|Divide)")

    def __init__(self, B, seq_lens, d_model, d_ff, heads, vocab, out_axis: str = "vocab"):
        self.B, self.lens = B, set(seq_lens)
        self.out_axis = out_axis
        self.sizes = {d_model: "features", d_ff: "hidden (d_ff)", d_model // heads: "head features",
                      vocab: out_axis}

    def last(self, n: int, keys: bool) -> str:
        if keys and n in self.lens:
            return "keys"
        return self.sizes.get(n) or ("seq" if n in self.lens else "dim")

    def __call__(self, shape: list[int], label: str = "", causal: bool = False) -> list[str]:
        n = len(shape)
        if n == 4:
            last = ("keys" if self.KEYS_LAST.match(label) else "head features" if self.HEAD_LAST.match(label)
                    else self.last(shape[3], keys=True))
            return ["batch", "heads", "queries", last]
        if n == 3:
            last = ("hidden (d_ff)" if self.HIDDEN.match(label) else self.out_axis if self.VOCAB.match(label)
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
            if shape[0] == self.B:
                return ["batch", self.sizes.get(shape[1], "dim")]
            return ["rows", "cols"]
        return ["dim"] * n


def sub_slice(t: torch.Tensor, names: list[str]) -> dict:
    """A small, display-ready slice of any tensor (singleton axes squeezed, last axis sampled)."""
    shp = list(t.shape)
    keep = [i for i, n in enumerate(shp) if n != 1] or [len(shp) - 1]
    tt = t.detach().reshape([shp[i] for i in keep])
    nm, full = [names[i] for i in keep], [shp[i] for i in keep]
    cut = {}
    for ax, name in enumerate(nm):          # long sequences: keep the first positions
        c = _seq_cut(full[ax]) if name in SEQ_AXES else None
        if c:
            cut[name] = full[ax]
            tt = tt.narrow(ax, 0, c)
            full[ax] = c
    idx = None
    cap = None if OPTS["detail"] else SUB_LAST.get(tt.dim())
    if cap and full[-1] > cap:
        idx = [i * full[-1] // cap for i in range(cap)]
        tt = tt.index_select(-1, torch.tensor(idx))
    kind = "bool" if t.dtype == torch.bool else "float" if t.is_floating_point() else "int"
    out = {"v": rl(tt.float().tolist()), "names": nm, "full": full, "idx": idx, "kind": kind}
    if cut:
        out["cut"] = cut
    return out


def short(code: str) -> str:
    code = code.strip()
    return code if len(code) <= 70 else code[:67] + "…"


def fmt(shape) -> str:
    return "(" + ", ".join(map(str, shape)) + ")"


def make_labeler(rules: list[tuple[str, str]]):
    """Human-readable name for the operation a line performs: first matching (pattern, label)."""
    def label(code: str, var: str | None) -> str:
        for pat, lab in rules:
            if pat in code:
                return lab
        return f"{var} = …" if var else short(code)
    return label


class Tracer:
    """Collects executed project lines and groups them into steps."""

    def __init__(self, root: Path, namer=None, label=None, glue: dict | None = None):
        self.root = root
        self.project_files = {str(p.resolve()): p.name for p in root.glob("*.py")}
        self.steps: list[dict] = []
        self.stack: list[dict] = []
        self.namer = namer
        self.label = label or make_labeler([])
        self.pending: dict = {}   # frame -> (line being executed, {local name: id(tensor)})
        self.lines: dict[str, list[str]] = {}
        # glue[(file, line)] = {"key", "inputs": [local names], "out": local name, "layer": fn(frame)}
        self.glue = glue or {}
        self.glue_active: dict = {}   # frame -> spec of the glue step it is running

    @staticmethod
    def glue_lines(root: Path, file: str, specs: list[dict]) -> dict:
        """Resolve glue specs ({"match": code substring, "key", …}) to {(file, line): spec}."""
        lines = (root / file).read_text(encoding="utf-8").splitlines()
        out = {}
        for spec in specs:
            hits = [i + 1 for i, l in enumerate(lines) if spec["match"] in l]
            if len(hits) != 1:
                raise SystemExit(f"[tracer] glue pattern {spec['match']!r} matches {len(hits)} lines in {file}")
            out[(file, hits[0])] = spec
        return out

    # ---- sub-steps ----------------------------------------------------------------
    def code_at(self, file: str, line: int) -> str:
        if file not in self.lines:
            self.lines[file] = (self.root / file).read_text(encoding="utf-8").splitlines()
        return self.lines[file][line - 1]

    def record_sub(self, t, file: str | None, line: int | None, label: str, var: str | None = None,
                   formula: str | None = None, force: bool = False, ops: dict | None = None) -> dict | None:
        """Add one tensor operation to the running step (deduplicated by tensor identity)."""
        if not self.stack or not isinstance(t, torch.Tensor):
            return None
        step = self.stack[-1]
        # Keep strong references so a freed temporary's id can't be reused and mistaken for a seen tensor.
        seen = step.setdefault("_seen", {})
        if seen.get(id(t)) is t and not force:
            return None
        seen[id(t)] = t
        sub = {"label": label, "var": var, "file": file, "line": line, "shape": list(t.shape),
               "dtype": str(t.dtype).replace("torch.", ""), "formula": formula}
        if file and line:
            sub["code"] = short(self.code_at(file, line))
        subs = step.setdefault("subs", [])
        if (OPTS["detail"] or step["layer"] in (None, 1)) and self.namer:   # full detail, or layer 1 only
            # In masked self-attention the first boolean mask converted is the causal (T, T) one.
            causal = (label.startswith("Boolean mask") and step["key"] == "dec_masked_attn"
                      and not any(x["label"].startswith("Boolean mask") for x in subs))
            names = step.get("_names", {}).get(label) or self.namer(list(t.shape), label, causal)
            sub["vals"] = sub_slice(t, names)
            if ops and OPTS["detail"]:
                sub["ops"] = ops
        subs.append(sub)
        return sub

    def _diff_locals(self, frame) -> None:
        prev = self.pending.get(frame)
        if prev is None or not self.stack:
            return
        line, snap = prev
        fname = self.project_files[frame.f_code.co_filename]
        code = self.code_at(fname, line)
        for name, val in frame.f_locals.items():
            if isinstance(val, torch.Tensor) and snap.get(name) != id(val):
                self.record_sub(val, fname, line, self.label(code, name), var=name)

    # ---- glue steps -------------------------------------------------------------
    def _glue_end(self, frame, line: int | None) -> None:
        spec = self.glue_active.get(frame)
        if spec is None:
            return
        fname = self.project_files[frame.f_code.co_filename]
        nxt = self.glue.get((fname, line)) if line else None
        if nxt is not None and nxt["key"] == spec["key"]:
            return                                     # the same glue step spans this line too
        del self.glue_active[frame]
        out = frame.f_locals.get(spec["out"])
        step = self.end(out)
        step["out_shape"] = shape(out)
        step["out_dtype"] = str(out.dtype).replace("torch.", "") if isinstance(out, torch.Tensor) else None
        step["out_vals"] = cube_slice(out)

    def _glue_begin(self, frame) -> None:
        if frame in self.glue_active:
            return
        fname = self.project_files[frame.f_code.co_filename]
        spec = self.glue.get((fname, frame.f_lineno))
        if spec is None:
            return
        loc = frame.f_locals
        inputs = [(n, loc.get(n)) for n in spec.get("inputs", [])]
        layer = spec["layer"](frame) if spec.get("layer") else None
        step = self.begin(spec["key"], layer, inputs, frame=frame)
        first = next((t for _, t in inputs if isinstance(t, torch.Tensor)), None)
        step["in_vals"] = cube_slice(first)
        # Every input's values, by name (e.g. a residual add shows both operands).
        step["in_all"] = {n: cube_slice(t) for n, t in inputs if isinstance(t, torch.Tensor)}
        self.glue_active[frame] = spec

    # ---- sys.settrace plumbing -------------------------------------------------
    def global_trace(self, frame, event, arg):
        if event == "call" and frame.f_code.co_filename in self.project_files:
            return self.local_trace
        return None

    def local_trace(self, frame, event, arg):
        if event in ("line", "return"):
            self._diff_locals(frame)  # the previous line of this frame just finished
            if self.glue:
                self._glue_end(frame, frame.f_lineno if event == "line" else None)
        if event == "line":
            if self.glue:
                self._glue_begin(frame)
            if self.stack:
                fname = self.project_files[frame.f_code.co_filename]
                step = self.stack[-1]
                step["exec"].setdefault(fname, set()).add(frame.f_lineno)
                step["first"].setdefault(fname, frame.f_lineno)  # first line run, in time order
            self.pending[frame] = (frame.f_lineno, {k: id(v) for k, v in frame.f_locals.items()
                                                     if isinstance(v, torch.Tensor)})
        elif event == "return":
            prev = self.pending.pop(frame, None)
            if prev and self.stack and isinstance(arg, torch.Tensor):
                fname = self.project_files[frame.f_code.co_filename]
                code = self.code_at(fname, prev[0])
                self.record_sub(arg, fname, prev[0], self.label(code, None), var="return")
        return self.local_trace

    def current_line(self) -> tuple[str | None, int | None]:
        """Innermost project frame's current line (the code that triggered a PyTorch op)."""
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename in self.project_files:
                return self.project_files[f.f_code.co_filename], f.f_lineno
            f = f.f_back
        return None, None

    # ---- step boundaries -------------------------------------------------------
    def _project_frames(self, f=None) -> list[dict]:
        frames = []
        f = f or sys._getframe(2)   # 0 = here, 1 = begin, 2 = its caller (a hook)
        while f is not None:
            if f.f_code.co_filename in self.project_files:
                frames.append({
                    "file": self.project_files[f.f_code.co_filename],
                    "line": f.f_lineno,
                    "func": getattr(f.f_code, "co_qualname", f.f_code.co_name),
                })
            f = f.f_back
        return frames[::-1]  # outermost first

    def begin(self, key: str, layer: int | None, inputs: list[tuple[str, object]], module=None, frame=None) -> dict:
        step = {
            "key": key,
            "layer": layer,
            "_module": module,
            "frames": self._project_frames(frame),
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

    # ---- module hooks ------------------------------------------------------------
    def hook(self, module, key, layer=None, on_post=None) -> None:
        """Make `module` a step: begin on its forward pre-hook, end on its forward hook."""
        def pre(mod, args, kwargs):
            k = key(mod) if callable(key) else key
            names = list(inspect.signature(mod.forward).parameters)
            named = list(zip(names, args)) + list(kwargs.items())
            step = self.begin(k, layer, named, module=mod)
            first = next((t for _, t in named if isinstance(t, torch.Tensor)), None)
            step["in_vals"] = cube_slice(first)

        def post(mod, args, kwargs, output):
            step = self.end(output)
            out = output[0] if isinstance(output, tuple) else output
            step["out_shape"] = shape(out)
            step["out_dtype"] = str(out.dtype).replace("torch.", "")
            step["out_vals"] = cube_slice(out)
            if OPTS["detail"] and isinstance(mod, torch.nn.Linear):   # operands for the matmul view
                step["ops"] = linear_ops(mod, args[0])
            if on_post:
                on_post(step, mod, args, output)

        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.register_forward_hook(post, with_kwargs=True)


def linear_ops(mod: torch.nn.Linear, x: torch.Tensor) -> dict:
    return {"x": rl(x), "W": rl(mod.weight), "b": rl(mod.bias) if mod.bias is not None else None}


def layernorm_subs(tracer: Tracer, step: dict, mod, x: torch.Tensor, out: torch.Tensor, post_norm: bool) -> None:
    """Decompose LayerNorm (computed in C++ by PyTorch) into its arithmetic, linked to the call site.

    Post-norm models (LN(x + Sublayer(x))) get a "Residual add" sub-step first; pre-norm models
    (x + Sublayer(LN(x))) add their residual later, in a step of its own.
    """
    call = step["frames"][-1] if step["frames"] else {"file": None, "line": None}
    file, line = call["file"], call["line"]
    tracer.stack.append(step)  # record into this (already finished) step
    try:
        if post_norm and not step["key"].endswith("final"):
            tracer.record_sub(x, file, line, "Residual add: x + Dropout(Sublayer(x))", var="x",
                              formula="x + Sublayer(x)", force=True)
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        xhat = (x - mu) / torch.sqrt(var + mod.eps)
        tracer.record_sub(mu, file, line, "Mean over the features", var="μ", formula="μ = mean(x)")
        tracer.record_sub(var, file, line, "Variance over the features", var="σ²", formula="σ² = mean((x − μ)²)")
        tracer.record_sub(xhat, file, line, "Normalize", var="x̂", formula="x̂ = (x − μ) / √(σ² + ε)")
        tracer.record_sub(out, file, line, "Scale and shift (learned γ, β)", var="y", formula="y = γ · x̂ + β",
                          ops={"gamma": rl(mod.weight), "beta": rl(mod.bias), "eps": mod.eps})
    finally:
        tracer.stack.pop()


def definition_line(model, name: str, project_files: dict) -> dict | None:
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
        file = project_files.get(str(Path(inspect.getsourcefile(type(owner))).resolve()))
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


def install_leaf_hooks(tracer: Tracer, model, extra=None) -> None:
    """Leaf PyTorch layers (Linear, Embedding, activations…) become sub-steps of the running step.

    `extra(mod, inp, out)` may return (label, formula, ops) for layer types a model adds (e.g. Conv2d).
    """
    names = {mod: n for n, mod in model.named_modules()}

    def leaf_hook(mod, inp, out):
        if not tracer.stack or mod is tracer.stack[-1].get("_module"):
            return
        if isinstance(mod, (torch.nn.Dropout, torch.nn.LayerNorm)):
            return  # dropout is the identity in eval mode; LayerNorm steps are decomposed separately
        file, line = tracer.current_line()
        attr = names[mod].split(".")[-1]
        attr = f"net[{attr}]" if attr.isdigit() else attr
        ops = None
        custom = extra(mod, inp, out) if extra else None
        if custom:
            label, formula, ops = custom
        elif isinstance(mod, torch.nn.Linear):
            label, formula = f"Linear {attr}: {mod.in_features} → {mod.out_features}", "x · Wᵀ + b"
            if OPTS["detail"]:   # operands for the cell-by-cell matmul view
                ops = linear_ops(mod, inp[0])
        elif isinstance(mod, torch.nn.Embedding):
            label, formula = "Embedding lookup: one table row per token id", "E[token]"
            if OPTS["detail"]:
                used = sorted({int(i) for i in inp[0].flatten().tolist()})
                ops = {"rows": {str(i): rl(mod.weight[i]) for i in used}}
        else:
            label, formula = f"{type(mod).__name__} activation", f"{type(mod).__name__.lower()}(x)"
            if OPTS["detail"]:
                ops = {"x": rl(inp[0])}
        sub = tracer.record_sub(out, file, line, label, var=attr, formula=formula, ops=ops)
        if sub is not None:
            d = definition_line(model, names[mod], tracer.project_files)
            if d:
                sub["def"] = d

    for mod in model.modules():
        if not list(mod.children()):
            mod.register_forward_hook(leaf_hook)


def finalize(steps: list[dict], step_map: dict, describe, nlayers) -> list[dict]:
    """Raw trace steps → the page's step records (title, description, primary file…)."""
    out = []
    for s in steps:
        block, comp, role = step_map[s["key"]]
        ins = s.get("in_shapes") or [[]]
        o = s.get("out_shape") or []
        title, desc = describe(s["key"], s["layer"], ins, o)
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
            "layer": s["layer"], "nlayers": nlayers(s["key"]),
            "title": title, "desc": desc, "frames": s["frames"], "exec": exec_, "primary": primary,
            "first": s.get("first", {}),
            "inputs": s.get("inputs", []) + s.get("scalar_inputs", []),
            "out_shape": o, "out_dtype": s.get("out_dtype"),
        }
        for k in ("attn", "mask", "topk", "topk_logits", "in_vals", "out_vals", "subs", "ops", "in_all", "extra"):
            if k in s:
                step[k] = s[k]
        out.append(step)
    return out
