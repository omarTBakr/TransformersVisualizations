"""
Build a model's architecture page and serve its ui/ folder on localhost, then open it.

Serving over http (instead of opening the file directly) lets pdf.js load your local copy
of the paper automatically. Each model's ui/serve.py calls `main(ui_dir)`.
"""

import argparse
import functools
import http.server
import importlib
import sys
import webbrowser
from pathlib import Path


def main(ui_dir: Path, doc: str = __doc__) -> None:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding architecture.html")
    parser.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    if not args.no_build:
        sys.path.insert(0, str(ui_dir))
        importlib.import_module("build_ui").main()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ui_dir))
    for port in range(args.port, args.port + 10):  # try the next ports if one is busy
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"No free port in {args.port}-{args.port + 9}")

    url = f"http://127.0.0.1:{port}/architecture.html"
    print(f"\nServing {ui_dir} at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
