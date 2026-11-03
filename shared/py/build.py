"""
Shared page builder: turns a model's components, simulations and notes into one
self-contained <model>/ui/architecture.html.

    shared/ui/template.html     the page shell (header, player, dock, panels, CSS, app logic)
    shared/ui/js/*.js           dock layout, isometric tensors, the operation-stage engine
    <model>/ui/js/model.js      that model's diagram, chips and aliases
    <model>/ui/js/stage-model.js that model's own operation stages and breadcrumb

Every script is inlined, so the page works from file:// as well as over http. Only the
git-ignored paper.local.js (the user's local copy of the paper) stays a separate file.
"""

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent
TEMPLATE = SHARED / "ui" / "template.html"
DATA_PLACEHOLDER = "/*__DATA__*/null"
TITLE_PLACEHOLDER = "/*__TITLE__*/"
# <script src> in the template → where the file comes from ("shared" or "model").
SCRIPTS = {
    "js/dock.js": "shared",
    "js/model.js": "model",
    "js/tensors.js": "shared",
    "js/stage-model.js": "model",
    "js/stage-core.js": "shared",
}


@dataclass
class Component:
    id: str
    label: str
    kind: str  # embed | pe | attn | ffn | norm | linear | softmax | stack | misc
    note: str
    refs: list[tuple[str, str | None]] = field(default_factory=list)  # (file, symbol)


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


def resolve(root: Path, file: str, symbol: str | None, cache: dict) -> dict:
    path = root / file
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

    return {"file": file, "symbol": symbol, "start": start, "end": end,
            "code": "\n".join(lines[start - 1: end])}


def components_and_files(root: Path, components: list[Component]) -> tuple[dict, dict]:
    """Component → resolved code ranges, and every project file → the blocks it implements."""
    cache: dict[str, str] = {}
    comps = {c.id: {"label": c.label, "kind": c.kind, "note": c.note,
                    "refs": [resolve(root, f, s, cache) for f, s in c.refs]} for c in components}
    project_files = sorted(p.name for p in root.glob("*.py")) + ["config.yml"]
    files = {}
    for name in project_files:
        text = cache.get(name) or (root / name).read_text(encoding="utf-8")
        files[name] = {
            "code": text,
            "lines": len(text.splitlines()),
            "blocks": [cid for cid, c in comps.items() if any(r["file"] == name for r in c["refs"])],
        }
    return comps, files


def check(data: dict, components: list[Component]) -> None:
    """Every implementation note and component must point at something that exists."""
    known = {c.id for c in components} | {s["key"] for sim in data["sims"].values() for s in sim["steps"]}
    bad = sorted({c for n in data["impl"] for c in n["comps"] if c not in known})
    if bad:
        raise SystemExit(f"[build_ui] Implementation notes reference unknown components: {bad}")
    missing = [c.id for c in components if c.id not in data["paper"]["entries"]]
    if missing:
        raise SystemExit(f"[build_ui] No paper entry for components: {missing} (see ui/paper.py)")


def write_page(ui_dir: Path, data: dict, title: str) -> Path:
    """Fill the shared template with DATA and the inlined scripts; write ui/architecture.html."""
    html = TEMPLATE.read_text(encoding="utf-8")
    for src, where in SCRIPTS.items():
        path = (SHARED / "ui" / src) if where == "shared" else (ui_dir / src)
        if not path.exists():
            raise SystemExit(f"[build_ui] Missing UI script: {path}")
        tag = f'<script src="{src}"></script>'
        if tag not in html:
            raise SystemExit(f"[build_ui] {tag} missing from shared/ui/template.html")
        code = path.read_text(encoding="utf-8")
        if "</script" in code.lower():
            raise SystemExit(f"[build_ui] {path} contains '</script', which would end the inlined <script>")
        html = html.replace(tag, f"<script>/* {where}: {src} */\n{code}\n</script>")
    if DATA_PLACEHOLDER not in html:
        raise SystemExit(f"[build_ui] Placeholder {DATA_PLACEHOLDER!r} missing from template.html")
    # Escape "</" so embedded source can never close the <script> tag.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace(TITLE_PLACEHOLDER, title).replace(DATA_PLACEHOLDER, payload)
    out = ui_dir / "architecture.html"
    out.write_text(html, encoding="utf-8")
    return out
