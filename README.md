# Transformers Visualizations

Interactive, step-by-step visualizations of Transformer architectures, driven by **real PyTorch forward passes**. Every tensor operation is shown with its real values, its arithmetic written out, and the exact line of code that produced it.

| Model | Status | Folder |
|---|---|---|
| Encoder–decoder (original Transformer, *Attention Is All You Need*) | ✅ available | [`encoder-decoder/`](encoder-decoder/) |
| Decoder-only (GPT-style) | planned | — |
| Encoder-only (BERT-style) | planned | — |

![Encoder–decoder visualization](encoder-decoder/demo/screenshots/01-overview.png)

## Try the encoder–decoder visualization

```bash
cd encoder-decoder
uv sync
uv run ui/serve.py
```

See the [encoder–decoder README](encoder-decoder/README.md) for the full feature list, how it works, and the demo material (video, carousel and screenshots).

Each model lives in its own folder, with its own model code, visualization and demo assets.
