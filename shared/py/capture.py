"""
Shared demo capture: regenerate a model's demo assets from its live page.

  <model>/demo/screenshots/*.png      feature screenshots (headless Chrome, 1920×1080)
  <model>/demo/linkedin-carousel.pdf  3-page LinkedIn document post (1080×1350), from demo/carousel.html
  <model>/demo/walkthrough.mp4        captioned tour (+ walkthrough.gif for the README)

Each model's demo/capture.py defines its shot list and calls `run(demo_dir, SHOTS_SPEC)`.
Needs google-chrome (or chromium) and ffmpeg on PATH.
"""

import functools
import http.server
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

W, H = 1920, 1080
FONT = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
DEMO = SHOTS = CARDS = Path(".")   # set by run()


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


def run(demo: Path, shots_spec: list[tuple[str, str, int, str]]) -> None:
    """Screenshots + cards + carousel PDF + captioned video for one model's demo/ folder."""
    global DEMO, SHOTS, CARDS
    DEMO, SHOTS, CARDS = demo, demo / "screenshots", demo / "cards"
    SHOTS.mkdir(exist_ok=True)
    server, port = serve(DEMO.parent)                  # serve <model>/ so ui/ and demo/ are reachable
    base = f"http://127.0.0.1:{port}"
    try:
        print("Screenshots:")
        for name, query, step, _ in shots_spec:
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
    make_video(shots_spec)


def make_video(shots_spec, seconds: float = 4.2, fade: float = 0.6) -> None:
    """Captioned slideshow: intro card → feature screenshots → outro card, with crossfades."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required for the video")
    frames = [(CARDS / "intro.png", None)] + [(SHOTS / f"{n}.png", c) for n, _, _, c in shots_spec] + [(CARDS / "outro.png", None)]
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
                        str(DEMO / "walkthrough.mp4")], check=True)
    # GIF for the README (GitHub plays GIFs inline, not MP4s)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(DEMO / "walkthrough.mp4"), "-vf",
                    "fps=6,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=sierra2_4a",
                    str(DEMO / "walkthrough.gif")], check=True)
    print("Video:\n  walkthrough.mp4\n  walkthrough.gif")
