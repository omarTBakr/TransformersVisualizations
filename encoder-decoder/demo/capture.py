"""
Regenerate the demo assets from the live app:

  demo/screenshots/*.png      feature screenshots (headless Chrome, 1920×1080)
  demo/linkedin-carousel.pdf  3-page LinkedIn document post (1080×1350)
  demo/feature-tour.mp4       ~1-minute captioned tour of all features

Usage (from encoder-decoder/):
    uv run ui/build_ui.py          # make sure the page is up to date
    uv run demo/capture.py         # needs google-chrome and ffmpeg on PATH

The paper's page viewer is deliberately not captured: full pages of the paper go beyond
the tables-and-figures permission, so they stay out of public demo material.
"""

import functools
import http.server
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

DEMO = Path(__file__).resolve().parent
UI = DEMO.parent / "ui"
SHOTS = DEMO / "screenshots"
CARDS = DEMO / "cards"
W, H = 1920, 1080
FONT = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"

# (file name, query string, step, caption)
SHOTS_SPEC = [
    ("01-overview", "config=tiny&stage=1", 5, "One page: the operation, the architecture and the code that runs it, all in sync"),
    ("02-embedding", "config=tiny&stage=2", 3, "Every token is looked up in a real embedding table"),
    ("03-positional-encoding", "config=tiny&stage=2", 4, "Positional encoding, position by position, with the sin / cos formula worked out"),
    ("04-matmul", "config=tiny&stage=1", 5, "Matrix multiplications: pick any cell to see its dot product, checked against the model"),
    ("05-attention-scores", "config=tiny&stage=5", 5, "Attention scores Q·Kᵀ / √d_k with the real numbers"),
    ("06-softmax", "config=tiny&sample=2&stage=7", 5, "Softmax written out in LaTeX; padded keys get −∞ and weight 0"),
    ("07-layernorm", "config=tiny&stage=4", 6, "LayerNorm: mean and variance along the feature axis, step by step"),
    ("08-relu", "config=tiny&stage=2", 7, "ReLU: negatives flagged, zeros in mint, the max starred"),
    ("09-small-heads", "config=small&stage=4", 5, "Small 9×9 preset: 3 attention heads, 2 + 2 layers, padding"),
    ("10-full-cubes", "config=full&stage=1", 4, "Full config.yml (d_model 512): isometric tensors with sampled values"),
    ("11-output-softmax", "config=tiny&stage=4", 21, "Next-token probabilities, with every exp and Σ shown"),
    ("12-read-the-code", "config=tiny&stage=6", 5, "Read the code: each stage outlines its exact line, plus details not in the paper"),
]


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # keep the capture output readable
            pass
    handler = functools.partial(Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def chrome(*args: str) -> None:
    exe = shutil.which("google-chrome") or shutil.which("chromium")
    if not exe:
        raise SystemExit("google-chrome (or chromium) is required")
    with tempfile.TemporaryDirectory() as profile:     # fresh profile: default layout, no saved state
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                        f"--user-data-dir={profile}", *args], check=True, capture_output=True)


def screenshot(url: str, out: Path, w: int = W, h: int = H, wait_ms: int = 9000) -> None:
    chrome(f"--window-size={w},{h}", f"--virtual-time-budget={wait_ms}", f"--screenshot={out}", url)
    print(f"  {out.relative_to(DEMO)}")


def main() -> None:
    SHOTS.mkdir(exist_ok=True)
    server, port = serve(DEMO.parent)                  # serve encoder-decoder/ so ui/ and demo/ are reachable
    base = f"http://127.0.0.1:{port}"
    try:
        print("Screenshots:")
        for name, query, step, _ in SHOTS_SPEC:
            screenshot(f"{base}/ui/architecture.html?{query}&theme=dark#s={step}", SHOTS / f"{name}.png")
        print("Cards:")
        for card in ("intro", "outro"):
            screenshot(f"{base}/demo/cards/{card}.html", CARDS / f"{card}.png", wait_ms=2000)
        print("Carousel:")
        chrome("--no-pdf-header-footer", f"--print-to-pdf={DEMO / 'linkedin-carousel.pdf'}", "--virtual-time-budget=4000",
               f"{base}/demo/carousel.html")
        print("  linkedin-carousel.pdf")
    finally:
        server.shutdown()
    make_video()


def make_video(seconds: float = 4.2, fade: float = 0.6) -> None:
    """Captioned slideshow: intro card → feature screenshots → outro card, with crossfades."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required for the video")
    frames = [(CARDS / "intro.png", None)] + [(SHOTS / f"{n}.png", c) for n, _, _, c in SHOTS_SPEC] + [(CARDS / "outro.png", None)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clips = []
        for k, (img, caption) in enumerate(frames):
            clip = tmp / f"c{k:02d}.mp4"
            vf = f"scale={W}:{H},format=yuv420p"
            if caption:
                cap = tmp / f"cap{k:02d}.txt"
                cap.write_text(caption, encoding="utf-8")
                vf += (f",drawbox=x=0:y=ih-110:w=iw:h=110:color=black@0.72:t=fill,"
                       f"drawtext=fontfile={FONT}:textfile={cap}:fontcolor=white:fontsize=40:x=(w-text_w)/2:y=h-75")
            dur = seconds + (1.2 if caption is None else 0)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", f"{dur}", "-i", str(img),
                            "-vf", vf, "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
            clips.append((clip, dur))
        # chain crossfades
        inputs, filt, last, offset = [], [], "[0:v]", 0.0
        for k, (clip, dur) in enumerate(clips):
            inputs += ["-i", str(clip)]
            if k == 0:
                offset = dur - fade
                continue
            out = f"[v{k}]"
            filt.append(f"{last}[{k}:v]xfade=transition=fade:duration={fade}:offset={offset:.2f}{out}")
            last, offset = out, offset + dur - fade
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt), "-map", last,
                        "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(DEMO / "feature-tour.mp4")], check=True)
    print("Video:\n  feature-tour.mp4")


if __name__ == "__main__":
    main()
