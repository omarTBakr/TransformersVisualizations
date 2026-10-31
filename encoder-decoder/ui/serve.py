"""
Build the architecture page and serve ui/ on localhost, then open it in the browser.

Serving over http (instead of opening the file directly) lets pdf.js load your local copy
of the paper automatically.

Usage:
    .venv/bin/python ui/serve.py            # build, serve on port 8765 and open the browser
    .venv/bin/python ui/serve.py --no-build # serve the existing architecture.html
"""

import argparse
import functools
import http.server
import sys
import webbrowser
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding architecture.html")
    parser.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    if not args.no_build:
        sys.path.insert(0, str(UI_DIR))
        import build_ui

        build_ui.main()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(UI_DIR))
    for port in range(args.port, args.port + 10):  # try the next ports if one is busy
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"No free port in {args.port}-{args.port + 9}")

    url = f"http://127.0.0.1:{port}/architecture.html"
    print(f"\nServing {UI_DIR} at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
