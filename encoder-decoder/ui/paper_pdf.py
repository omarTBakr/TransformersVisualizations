"""
Locate passages in the user's local copy of the paper so the UI can draw highlight boxes
on top of the real PDF (rendered in the browser by pdf.js).

Requires poppler-utils (`pdftotext`). If the PDF or the tool is missing, everything
degrades gracefully: the UI simply hides the paper viewer.
"""

import base64
import html
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent

_WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>'
)


def _norm(text: str) -> str:
    """Lowercase alphanumerics only, so hyphenation, quotes and spacing don't matter."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


@dataclass
class Word:
    page: int
    line: int  # global line id
    box: tuple[float, float, float, float]


class PaperPDF:
    def __init__(self, pdf: Path):
        self.pdf = pdf
        self.words: list[Word] = []
        self.page_size: dict[int, tuple[float, float]] = {}
        self._chars = ""      # normalized text of the whole document
        self._owner: list[int] = []  # char index -> word index
        self._extract()

    # ---- text extraction --------------------------------------------------------
    def _extract(self) -> None:
        out = subprocess.run(
            ["pdftotext", "-bbox-layout", str(self.pdf), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
        chars, owner = [], []
        line_id = 0
        for p_no, page in enumerate(re.split(r"<page ", out)[1:], start=1):
            m = re.match(r'width="([\d.]+)" height="([\d.]+)"', page)
            self.page_size[p_no] = (float(m.group(1)), float(m.group(2)))
            for line in re.findall(r"<line .*?</line>", page, flags=re.S):
                line_id += 1
                for x0, y0, x1, y1, txt in _WORD_RE.findall(line):
                    idx = len(self.words)
                    self.words.append(Word(p_no, line_id, (float(x0), float(y0), float(x1), float(y1))))
                    n = _norm(html.unescape(txt))
                    chars.append(n)
                    owner.extend([idx] * len(n))
        self._chars = "".join(chars)
        self._owner = owner

    # ---- search -----------------------------------------------------------------
    def find(self, start: str, end: str | None = None, after: int = 0) -> tuple[int, int] | None:
        """Return (first_word, last_word) of the span from `start` through `end`."""
        s = _norm(start)
        i = self._chars.find(s, after)
        if i < 0:
            return None
        if end is None:
            j_end = i + len(s) - 1
        else:
            e = _norm(end)
            j = self._chars.find(e, i)
            if j < 0:
                return None
            j_end = j + len(e) - 1
        return self._owner[i], self._owner[j_end]

    def page_of(self, phrase: str) -> int | None:
        span = self.find(phrase)
        return self.words[span[0]].page if span else None

    def highlight(self, start: str, end: str | None = None) -> list[dict] | None:
        """Per-page highlight rectangles (in PDF points) for a passage, one per text line."""
        span = self.find(start, end)
        if span is None:
            return None
        lines: dict[int, list[float]] = {}
        page_of_line: dict[int, int] = {}
        for w in self.words[span[0]: span[1] + 1]:
            b = lines.setdefault(w.line, [w.box[0], w.box[1], w.box[2], w.box[3]])
            b[0], b[1] = min(b[0], w.box[0]), min(b[1], w.box[1])
            b[2], b[3] = max(b[2], w.box[2]), max(b[3], w.box[3])
            page_of_line[w.line] = w.page
        pages: dict[int, list[list[float]]] = {}
        for line, b in lines.items():
            pages.setdefault(page_of_line[line], []).append([round(v, 1) for v in b])
        return [{"page": p, "rects": r} for p, r in sorted(pages.items())]

    def page_sizes(self) -> dict[int, list[float]]:
        return {p: [w, h] for p, (w, h) in self.page_size.items()}


LOCAL_JS = UI_DIR / "paper.local.js"


def write_local_js(pdf: Path | None) -> None:
    """
    Embed the PDF as base64 in ui/paper.local.js so the viewer works when architecture.html
    is opened straight from disk (file://), where browsers block reading local files.
    The file is git-ignored: it's a copy of the paper and must not be redistributed.
    """
    if pdf is None:
        LOCAL_JS.unlink(missing_ok=True)  # don't leave a stale copy behind
        return
    if LOCAL_JS.exists() and LOCAL_JS.stat().st_mtime >= pdf.stat().st_mtime:
        return  # already up to date
    encoded = base64.b64encode(pdf.read_bytes()).decode("ascii")
    LOCAL_JS.write_text(f'window.PAPER_PDF_B64 = "{encoded}";\n', encoding="ascii")


def open_paper() -> PaperPDF | None:
    """Find the paper PDF in ui/ and parse it, or return None if unavailable."""
    candidates = sorted(UI_DIR.glob("*.pdf"))
    pdf = next((p for p in candidates if "attention" in p.name.lower()), candidates[0] if candidates else None)
    write_local_js(pdf)
    if not shutil.which("pdftotext"):
        print("[build_ui] poppler-utils not found: paper viewer disabled.")
        return None
    if pdf is None:
        print("[build_ui] No paper PDF in ui/: paper viewer disabled.")
        return None
    return PaperPDF(pdf)
