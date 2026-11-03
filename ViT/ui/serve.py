"""
Build the architecture page and serve ui/ on localhost, then open it in the browser.

Serving over http (instead of opening the file directly) lets pdf.js load your local copy
of the paper automatically.

Usage:
    uv run ui/serve.py            # build, serve on port 8765 and open the browser
    uv run ui/serve.py --no-build # serve the existing architecture.html
"""

import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR.parent.parent / "shared" / "py"))

import serve  # noqa: E402  (shared/py/serve.py)

if __name__ == "__main__":
    serve.main(UI_DIR, __doc__)
