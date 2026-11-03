"""
What "An Image Is Worth 16x16 Words" (Dosovitskiy et al., ICLR 2021) says about each block,
for the Paper panel.

The paper carries no reproduction-permission notice, so it is treated as all rights reserved:
every note below is an original paraphrase (no quotations), the equations are standard
notation, and the diagram is drawn from scratch. Highlights only *point* at passages in the
user's own local copy of the PDF, which is git-ignored and never redistributed.

Each entry:
  refs   – (section, title, ar5iv anchor) links
  eq     – optional equation (HTML)
  points – paraphrased notes; the ones in `focus` are highlighted for this block / step
  code   – how this repository relates to the paper for this part
"""

from pathlib import Path

PAPER_TITLE = "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)"
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2010.11929"
PAPER_PDF = "https://arxiv.org/pdf/2010.11929"

CITATION = ("Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, "
            "Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. "
            "“An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” ICLR 2021. arXiv:2010.11929.")

DISCLAIMER = (
    "This is an independent educational project. It is not affiliated with, endorsed by, or sponsored by "
    "Google or the paper's authors. The paper and its contents are © the authors. The paper has no "
    "reproduction-permission notice, so nothing from it is reproduced here: the notes are original "
    "paraphrases written for this project, the equations are standard mathematical notation, and the "
    "architecture diagram is an original drawing that follows the idea of the paper's Figure 1. "
    "Read the original paper for the authoritative description. Nothing here is legal advice."
)

LOCAL_NOTICE = (
    "The viewer displays your own copy of the PDF from ui/, rendered in your browser. The build also "
    "embeds that copy in ui/paper.local.js so the page works when opened from disk. Both the PDF and "
    "ui/paper.local.js are git-ignored. Please don't commit or share them."
)

S3 = ("3", "Method", "S3")
F1 = ("Fig. 1", "Model overview", "S3.F1")
S31 = ("3.1", "Vision Transformer (ViT)", "S3.SS1")
S32 = ("3.2", "Fine-tuning and higher resolution", "S3.SS2")
S41 = ("4.1", "Setup", "S4.SS1")
T1 = ("Table 1", "Model variants", "S4.T1")
APPA = ("A", "Multihead self-attention", "A1")
D3 = ("D.3", "Head type and class token", "A4.SS3")
D4 = ("D.4", "Positional embedding", "A4.SS4")

EQ1 = ("z<sub>0</sub> = [x<sub>class</sub>; x<sub>p</sub><sup>1</sup>E; x<sub>p</sub><sup>2</sup>E; …; x<sub>p</sub><sup>N</sup>E] + E<sub>pos</sub>,"
       "&nbsp; E ∈ ℝ<sup>(P²·C)×D</sup>, E<sub>pos</sub> ∈ ℝ<sup>(N+1)×D</sup>")
EQ2 = "z′<sub>ℓ</sub> = MSA(LN(z<sub>ℓ−1</sub>)) + z<sub>ℓ−1</sub>"
EQ3 = "z<sub>ℓ</sub> = MLP(LN(z′<sub>ℓ</sub>)) + z′<sub>ℓ</sub>"
EQ4 = "y = LN(z<sub>L</sub><sup>0</sup>)"

PATCH_POINTS = [
    "The standard Transformer takes a 1-D sequence of token vectors, so the image is reshaped into a sequence of "
    "N = HW/P² flattened patches, each holding P²·C values.",
    "N, the number of patches, is also the length of the sequence the Transformer sees.",
    "A trainable linear projection E maps every flattened patch to the model width D; its outputs are called the "
    "patch embeddings (Eq. 1).",
    "Hybrid variant: instead of raw pixels, the patches can be cut from a CNN feature map, and E is applied to those; "
    "with 1×1 patches this just flattens the feature map.",
]
CLS_POINTS = [
    "Like BERT's [class] token, a learnable vector is prepended to the sequence of patch embeddings.",
    "Its state at the encoder output serves as the image representation y (Eq. 4), and the classification head reads only that.",
    "Keeping the extra token follows the original text Transformer; pooling the patch tokens can work too if the "
    "learning rate is tuned (Appendix D.3).",
]
POS_POINTS = [
    "Position embeddings are added to the patch embeddings so the model knows where each patch came from.",
    "The paper uses standard learnable 1-D position embeddings (raster order); 2-D-aware variants gave no "
    "significant gain (Appendix D.4).",
    "In the ablation, having any positional information mattered a lot, but the kind of encoding made little difference.",
    "At higher fine-tuning resolution the sequence gets longer, so the learned position embeddings are interpolated in 2-D.",
]
ENC_POINTS = [
    "The encoder alternates multi-head self-attention (MSA) and MLP blocks (Eq. 2, 3).",
    "LayerNorm is applied before every block, and a residual connection after every block.",
    "The MLP has two layers with a GELU non-linearity.",
]
ATTN_POINTS = [
    "Self-attention is global: every patch can attend to every other patch from the first layer on.",
    "Multi-head self-attention runs k attention heads in parallel and projects their concatenation back to D "
    "(Appendix A); per-head width is D/k so the cost stays constant.",
    "ViT has much less image-specific inductive bias than a CNN: only the MLPs are local, and the 2-D structure is "
    "used just when cutting patches and when adjusting position embeddings.",
]
HEAD_POINTS = [
    "The classification head is attached to the [class] token's final state.",
    "It is an MLP with one hidden layer during pre-training and a single linear layer during fine-tuning.",
    "For fine-tuning, the pre-trained head is replaced by a zero-initialized D × K layer, K being the number of classes.",
]


def _entries(m, t) -> dict:
    hd = m.d_model // m.n_heads
    return {
        "vit": {
            "refs": [S3, F1, S31],
            "eq": "<br>".join([EQ1, EQ2, EQ3, EQ4]),
            "points": [
                "The model design follows the original Transformer as closely as possible, so NLP Transformer "
                "implementations can be reused almost unchanged.",
                *PATCH_POINTS[:1], *ENC_POINTS[:2], CLS_POINTS[1],
            ],
            "focus": [0, 1],
            "code": "transformer.py wires ImageEmbedding → Encoder → [CLS] → head, as in Eq. 1–4.",
        },
        "patch_embedding": {
            "refs": [S31, F1],
            "eq": EQ1,
            "points": PATCH_POINTS,
            "focus": [0, 2],
            "code": f"ImageEmbedding.py computes x_p·E for all N = {m.n_patches} patches at once with a Conv2d "
                    f"(kernel = stride = {m.patch_size}): that is exactly E, reshaped as a convolution kernel. "
                    "It is not the paper's hybrid CNN; it is still a single linear map of raw pixels.",
        },
        "extra_linear": {
            "refs": [S31],
            "eq": EQ1,
            "points": PATCH_POINTS[2:3] + ["Eq. 1 has exactly one projection, E, between the flattened patches and the "
                                          "patch embeddings."],
            "focus": [1],
            "code": "Not in the paper. ImageEmbedding.py adds Linear(D → D) after the projection. Two linear maps compose "
                    "into one (x·E·W + b′ is still linear in x), so this adds D² + D parameters but no expressive power.",
        },
        "cls_token": {
            "refs": [S31, D3],
            "eq": EQ1,
            "points": CLS_POINTS,
            "focus": [0],
            "code": "ImageEmbedding.py keeps one (1, 1, D) parameter, initialised from N(0, 0.02²), and expands it over "
                    "the batch without copying.",
        },
        "positional_encoding": {
            "refs": [S31, D4, S32],
            "eq": EQ1,
            "points": POS_POINTS,
            "focus": [0, 1],
            "code": f"Deviation: config.yml uses positional_encoding = {m.positional_encoding!r}. The paper learns its "
                    "position embeddings; this code adds the fixed sinusoids of the original Transformer (switch to "
                    "'learned' in config.yml to match the paper).",
        },
        "encoder_stack": {
            "refs": [S31, T1],
            "eq": EQ4,
            "points": ENC_POINTS[:2] + CLS_POINTS[1:2],
            "focus": [2],
            "code": f"encoder.py stacks L = {m.n_encoder_layers} pre-norm layers and ends with the final LayerNorm of Eq. 4.",
        },
        "encoder_layer": {
            "refs": [S31],
            "eq": EQ2 + "<br>" + EQ3,
            "points": ENC_POINTS,
            "focus": [0, 1],
            "code": "EncoderLayer.forward is Eq. 2 and Eq. 3, line for line.",
        },
        "enc_norm": {
            "refs": [S31],
            "eq": EQ2 + "<br>" + EQ3,
            "points": ENC_POINTS[1:2] + ["This is the pre-norm arrangement; the original Transformer normalized after "
                                         "the residual instead."],
            "focus": [0],
            "code": f"norm1 / norm2 in EncoderLayer, eps = {m.layer_norm_eps:g}. The residual adds back the "
                    "un-normalized input.",
        },
        "enc_self_attn": {
            "refs": [APPA, S31],
            "eq": EQ2,
            "points": ATTN_POINTS,
            "focus": [0, 1],
            "code": f"multiheadAttention.py: k = {m.n_heads} heads of D/k = {hd} dims, no masks (every image has the same "
                    "number of patches, so there is no padding).",
        },
        "enc_residual": {
            "refs": [S31],
            "eq": EQ2 + "<br>" + EQ3,
            "points": ENC_POINTS[1:2],
            "focus": [0],
            "code": "x = x + dropout(sublayer(norm(x))) in EncoderLayer.forward.",
        },
        "enc_ffn": {
            "refs": [S31],
            "eq": EQ3,
            "points": ENC_POINTS[2:3] + ["Only the MLP layers are local and translation-equivariant: they act on each "
                                         "token separately."],
            "focus": [0],
            "code": f"feedForward.py: Linear {m.d_model} → {m.d_ff}, {m.activation.upper()}, Dropout, Linear {m.d_ff} → {m.d_model}.",
        },
        "cls_select": {
            "refs": [S31, D3],
            "eq": EQ4,
            "points": CLS_POINTS[1:2] + HEAD_POINTS[:1],
            "focus": [0],
            "code": "transformer.py takes x[:, 0, :] after the encoder's final LayerNorm.",
        },
        "head": {
            "refs": [S31, S32],
            "points": HEAD_POINTS,
            "focus": [1],
            "code": f"transformer.py uses the fine-tuning form: one Linear D = {m.d_model} → K = {m.num_classes}.",
        },
        "softmax": {
            "refs": [S31],
            "points": ["The head outputs one score per class; a softmax turns the scores into class probabilities."],
            "focus": [],
            "code": "VisionTransformer.forward returns logits; softmax belongs to the loss (or to inference).",
        },
    }


def _comparison(m, t) -> list[dict]:
    """ViT-Base/16 (Table 1 and §4.1) vs the values in config.yml."""
    rows = [
        ("image size", 224, m.image_size),
        ("patch size P", 16, m.patch_size),
        ("patches N", 196, m.n_patches),
        ("layers L", 12, m.n_encoder_layers),
        ("hidden size D", 768, m.d_model),
        ("MLP size", 3072, m.d_ff),
        ("heads", 12, m.n_heads),
        ("activation", "gelu", m.activation),
        ("position embeddings", "learned", m.positional_encoding),
        ("norm placement", "pre-norm", "pre-norm"),
        ("extra Linear after E", "no", "yes"),
        ("Adam β1, β2", "0.9, 0.999", ", ".join(str(b) for b in t.optimizer.betas)),
        ("weight decay", 0.1, t.optimizer.weight_decay),
        ("batch size", 4096, t.batch_size),
        ("LR warmup", 10000, t.scheduler.warmup_steps),
    ]
    return [{"name": n, "paper": str(p), "code": str(c), "same": str(p) == str(c)} for n, p, c in rows]


# Where each note lives in the paper: (start phrase, end phrase), used only to locate the passage in
# the user's local PDF. Keys are the opening words of the note. Notes without an anchor aren't highlighted.
ANCHORS = {
    "The model design follows": ("In model design we follow the original Transformer", "almost out of the box"),
    "The standard Transformer takes": ("The standard Transformer receives as input", "resulting number of patches"),
    "N, the number of patches": ("is the resulting number of patches", "input sequence length for the"),
    "A trainable linear projection": ("flatten the patches and map to D dimensions", "as the patch embeddings"),
    "Eq. 1 has exactly one": ("flatten the patches and map to D dimensions", "trainable linear projection"),
    "Hybrid variant": ("Hybrid Architecture. As an alternative to raw image patches", "projecting to the Transformer dimension"),
    "Like BERT's": ("Similar to BERT", "sequence of embedded patches"),
    "Its state at the encoder output": ("whose state at the output of the Transformer encoder", "image representation y"),
    "Keeping the extra token": ("In order to stay as close as possible to the original Transformer model", "class token"),
    "Position embeddings are added": ("Position embeddings are added to the patch embeddings", "retain positional information"),
    "The paper uses standard learnable": ("We use standard learnable 1D position embeddings", "(Appendix D.4)"),
    "At higher fine-tuning resolution": ("When feeding images of higher resolution", "location in the original image"),
    "The encoder alternates": ("The Transformer encoder (Vaswani et al., 2017) consists of", "MLP blocks (Eq. 2, 3)"),
    "LayerNorm is applied before": ("Layernorm (LN) is applied before every block", "after every block"),
    "The MLP has two layers": ("The MLP contains two layers with a GELU", None),
    "ViT has much less": ("We note that Vision Transformer has much less image-specific inductive bias", "self-attention layers are global"),
    "Self-attention is global": ("while the self-attention layers are global", None),
    "Only the MLP layers are local": ("In ViT, only MLP layers are local", "self-attention layers are global"),
    "The classification head is attached": ("a classification head is attached to", None),
    "It is an MLP with one hidden layer": ("The classification head is implemented by a MLP", "linear layer at fine-tuning time"),
    "For fine-tuning, the pre-trained head": ("we remove the pre-trained prediction head", "number of downstream classes"),
}

# Phrases that locate each section heading / figure caption (for the "open section" page).
SECTION_ANCHORS = {
    "S3": "3 Method", "S3.F1": "Figure 1: Model overview", "S3.SS1": "3.1 Vision Transformer (ViT)",
    "S3.SS2": "3.2 Fine-tuning and Higher Resolution", "S4.SS1": "4.1 Setup",
    "S4.T1": "Table 1: Details of Vision Transformer model variants", "A1": "A Multihead Self-attention",
    "A4.SS3": "D.3 Head Type and Class Token", "A4.SS4": "D.4 Positional Embedding",
}


def _attach_pdf(entries: dict, pdf) -> dict | None:
    """Add page numbers and highlight boxes from the local PDF; return page info."""
    done = set()
    for e in entries.values():
        if id(e) in done:
            continue
        done.add(id(e))
        pts = []
        for text in e["points"]:
            span = next((v for k, v in ANCHORS.items() if text.startswith(k)), None) if pdf else None
            hl = pdf.highlight(*span) if span else None
            if span and hl is None:
                raise SystemExit(f"[build_ui] Paper passage not found in the PDF: {span[0]!r}")
            pts.append({"text": text, "hl": hl})
        e["points"] = pts
        for r in e["refs"]:
            anchor_id = r["url"].rsplit("#", 1)[-1]
            r["page"] = pdf.page_of(SECTION_ANCHORS[anchor_id]) if pdf and anchor_id in SECTION_ANCHORS else None
    if pdf is None:
        return None
    sizes = pdf.page_sizes()
    return {"file": pdf.pdf.name, "npages": len(sizes), "sizes": sizes}


def paper_data(settings) -> dict:
    m, t = settings.model, settings.training
    entries = _entries(m, t)
    entries["image_embedding"] = entries["patch_embedding"]
    entries["main"] = entries["vit"]
    entries["config"] = {
        "refs": [T1, S41],
        "points": ["The table compares ViT-Base/16 (Table 1, §4.1) with the values in config.yml."],
        "focus": [],
        "code": "Rows that differ from the paper are highlighted.",
        "table": _comparison(m, t),
    }
    for e in {id(e): e for e in entries.values()}.values():
        e["refs"] = [{"sec": s, "title": ti, "url": f"{PAPER_URL}#{a}"} for s, ti, a in e["refs"]]

    from paper_pdf import open_paper  # noqa: E402

    local = _attach_pdf(entries, open_paper(Path(__file__).resolve().parent, "image is worth"))
    return {
        "title": PAPER_TITLE, "url": PAPER_URL, "pdf": PAPER_PDF, "entries": entries,
        "citation": CITATION, "notice": None, "disclaimer": DISCLAIMER,
        "local": local, "local_notice": LOCAL_NOTICE,
    }
