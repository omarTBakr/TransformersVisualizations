"""
Links between each diagram block / simulation step and the relevant part of
"Attention Is All You Need" (Vaswani et al., 2017, arXiv:1706.03762).

The notes are short paraphrases written for this project, not quotations. Equations are
given as math. Each entry links to the section on ar5iv so the original text is one click away.

Entry fields:
  refs   – list of (section number, title, ar5iv anchor id)
  eq     – optional HTML equation(s)
  points – paraphrased key ideas from those sections
  focus  – indices into `points` to highlight for this block / step
  code   – how this repository relates to the paper for this part
"""

from pathlib import Path

PAPER_TITLE = "Attention Is All You Need (Vaswani et al., 2017)"
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/1706.03762"
PAPER_PDF = "https://arxiv.org/pdf/1706.03762"

CITATION = ("Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, "
            "Łukasz Kaiser, Illia Polosukhin. “Attention Is All You Need.” Advances in Neural Information "
            "Processing Systems 30 (NeurIPS 2017). arXiv:1706.03762.")

# Permission notice printed on the first page of the arXiv version of the paper.
GOOGLE_NOTICE = ("Provided proper attribution is provided, Google hereby grants permission to reproduce "
                 "the tables and figures in this paper solely for use in journalistic or scholarly works.")

DISCLAIMER = (
    "This is an independent educational project. It is not affiliated with, endorsed by, or sponsored by "
    "Google or the paper's authors. The paper and its contents are © 2017 the authors / Google. "
    "The notes on this page are original paraphrases written for this project, not quotations from the paper; "
    "the equations are standard mathematical notation from the paper, and the architecture diagram is an "
    "original drawing that follows the layout of the paper's Figure 1. Google's permission above covers "
    "tables and figures for journalistic or scholarly use only; it doesn't extend to the paper's text. "
    "Read the original paper for the authoritative description. Nothing here is legal advice."
)

LOCAL_NOTICE = (
    "The viewer displays your own copy of the PDF from ui/, rendered in your browser. The build also "
    "embeds that copy in ui/paper.local.js so the page works when opened from disk. Both the PDF and "
    "ui/paper.local.js are git-ignored. Please don't commit or share them, since the full paper goes "
    "beyond the tables-and-figures permission above."
)

S3 = ("3", "Model Architecture", "S3")
F1 = ("Fig. 1", "The Transformer architecture", "S3.F1")
S31 = ("3.1", "Encoder and Decoder Stacks", "S3.SS1")
S321 = ("3.2.1", "Scaled Dot-Product Attention", "S3.SS2.SSS1")
S322 = ("3.2.2", "Multi-Head Attention", "S3.SS2.SSS2")
S323 = ("3.2.3", "Applications of Attention in our Model", "S3.SS2.SSS3")
F2 = ("Fig. 2", "Scaled dot-product and multi-head attention", "S3.F2")
S33 = ("3.3", "Position-wise Feed-Forward Networks", "S3.SS3")
S34 = ("3.4", "Embeddings and Softmax", "S3.SS4")
S35 = ("3.5", "Positional Encoding", "S3.SS5")
S51 = ("5.1", "Training Data and Batching", "S5.SS1")
S54 = ("5.4", "Regularization", "S5.SS4")
S62 = ("6.2", "Model Variations", "S6.SS2")

EQ_ATTN = "Attention(Q, K, V) = softmax( QK<sup>T</sup> / √d<sub>k</sub> ) V"
EQ_MHA = ("MultiHead(Q, K, V) = Concat(head<sub>1</sub>, …, head<sub>h</sub>) W<sup>O</sup><br>"
          "head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)")
EQ_FFN = "FFN(x) = max(0, xW<sub>1</sub> + b<sub>1</sub>) W<sub>2</sub> + b<sub>2</sub>"
EQ_PE = ("PE<sub>(pos, 2i)</sub> = sin( pos / 10000<sup>2i / d<sub>model</sub></sup> )<br>"
         "PE<sub>(pos, 2i+1)</sub> = cos( pos / 10000<sup>2i / d<sub>model</sub></sup> )")
EQ_NORM = "output = LayerNorm( x + Sublayer(x) )"

ATTN_POINTS = [
    "Scaled dot-product attention takes the dot product of each query with every key, divides it by √d_k, "
    "and applies a softmax to get the weights used to average the values.",
    "The √d_k scaling matters: without it, large dot products push the softmax into regions with tiny gradients.",
    "Multi-head attention projects Q, K and V h times with different learned matrices, runs attention on each "
    "projection in parallel, then concatenates the heads and projects them back.",
    "The base model uses h = 8 heads with d_k = d_v = d_model / h = 64, so the total cost is close to one full-width head.",
]


def _entries(m, t) -> dict:
    hd = m.d_model // m.n_heads
    attn_code = (f"multiheadAttention.py implements Eq. 1 and the multi-head projection with "
                 f"h = {m.n_heads} heads of {hd} dims (d_model = {m.d_model}).")
    return {
        "transformer": {
            "refs": [S3, F1],
            "points": [
                "The model is an encoder–decoder: the encoder turns the input tokens into a sequence of continuous "
                "representations, and the decoder generates the output one token at a time.",
                "Generation is auto-regressive: the tokens produced so far are fed back as extra input for the next step.",
                "Both halves are built only from stacked self-attention and position-wise fully connected layers, "
                "with no recurrence or convolution.",
            ],
            "focus": [0, 2],
            "code": "transformer.py wires these pieces together exactly as in Figure 1.",
        },
        "input_embedding": {
            "refs": [S34],
            "points": [
                "Learned embeddings map input and output tokens to vectors of size d_model.",
                "The paper shares one weight matrix between both embedding layers and the pre-softmax linear layer.",
                "Inside the embedding layers, the weights are multiplied by √d_model.",
            ],
            "focus": [0, 2],
            "code": f"inputEmbedding.py scales by √{m.d_model} ≈ {m.d_model ** 0.5:.1f}. The source embedding has its "
                    "own matrix here; the paper's three-way sharing assumes one shared vocabulary.",
        },
        "output_embedding": {
            "refs": [S34, S31],
            "points": [
                "Learned embeddings map input and output tokens to vectors of size d_model.",
                "The paper shares one weight matrix between both embedding layers and the pre-softmax linear layer.",
                "Inside the embedding layers, the weights are multiplied by √d_model.",
                "The decoder's input is the output sequence shifted right by one position, so position i only "
                "has the tokens before i.",
            ],
            "focus": [1, 3],
            "code": "outputEmbedding.py's weight is tied to the generator when share_embeddings is "
                    f"{'on (it is)' if m.share_embeddings else 'on (it is currently off)'}.",
        },
        "positional_encoding": {
            "refs": [S35, S54, S62],
            "eq": EQ_PE,
            "points": [
                "With no recurrence or convolution, the model needs extra information about token order.",
                "Positional encodings of size d_model are added to the embeddings at the bottom of both the "
                "encoder and decoder stacks.",
                "Each dimension is a sinusoid; the wavelengths form a geometric progression from 2π to 10000·2π.",
                "Sinusoids were chosen because the encoding of position pos + k is a linear function of the "
                "encoding at pos, which should make relative positions easy to attend to.",
                "Learned positional embeddings gave nearly identical results (Table 3, row E); sinusoids may "
                "extrapolate to longer sequences.",
                "Dropout (P_drop = 0.1) is applied to the sum of embeddings and positional encodings.",
            ],
            "focus": [1, 2],
            "code": f"positionalEncoding.py supports both options; config.yml selects "
                    f"'{m.positional_encoding}'. Dropout here is {m.dropout}.",
        },
        "enc_self_attn": {
            "refs": [S321, S322, S323, F2],
            "eq": EQ_ATTN + "<br>" + EQ_MHA,
            "points": ATTN_POINTS + [
                "In encoder self-attention, queries, keys and values all come from the previous encoder layer's "
                "output, so every position can attend to every position of the input.",
            ],
            "focus": [0, 4],
            "code": attn_code + " Padding masks are an implementation detail the paper doesn't discuss.",
        },
        "dec_masked_attn": {
            "refs": [S323, S31, S321],
            "eq": EQ_ATTN,
            "points": ATTN_POINTS + [
                "Decoder self-attention lets each position attend to positions up to and including itself.",
                "To keep the model auto-regressive, illegal (future) connections are masked by setting them "
                "to −∞ before the softmax.",
            ],
            "focus": [4, 5],
            "code": "Transformer.decode builds the causal mask; MultiHeadAttention adds −∞ at the masked positions.",
        },
        "causal_mask": {
            "refs": [S31, S323],
            "points": [
                "The decoder's self-attention is modified so that a position can't attend to later positions.",
                "Together with the one-position shift of the output embeddings, this means the prediction for "
                "position i depends only on the known outputs before i.",
                "The masking is done inside scaled dot-product attention by setting illegal connections to −∞ "
                "before the softmax.",
            ],
            "focus": [0, 1],
            "code": "Transformer.generate_square_subsequent_mask returns a (T, T) boolean matrix where True = blocked.",
        },
        "dec_cross_attn": {
            "refs": [S323, S31],
            "eq": EQ_ATTN,
            "points": ATTN_POINTS + [
                "In encoder–decoder attention, the queries come from the previous decoder layer, while the keys "
                "and values come from the encoder's output (the memory).",
                "This lets every decoder position attend over all input positions, like the attention used in "
                "earlier sequence-to-sequence models.",
            ],
            "focus": [4, 5],
            "code": "DecoderLayer.forward calls cross_attn(x, enc_output, enc_output) with the source padding mask.",
        },
        "enc_add_norm": {
            "refs": [S31, S54],
            "eq": EQ_NORM,
            "points": [
                "Each sub-layer has a residual connection around it, followed by layer normalization.",
                "All sub-layers and embedding layers output d_model = 512 dimensions so that the residual "
                "additions line up.",
                "Dropout is applied to each sub-layer's output before it's added to the input and normalized "
                "(P_drop = 0.1).",
            ],
            "focus": [0, 2],
            "code": "This code uses the same post-norm order: norm(x + dropout(sublayer(x))).",
        },
        "dec_add_norm": {
            "refs": [S31, S54],
            "eq": EQ_NORM,
            "points": [
                "Like the encoder, every decoder sub-layer (there are three) is wrapped in a residual connection "
                "followed by layer normalization.",
                "All sub-layers and embedding layers output d_model = 512 dimensions so that the residual "
                "additions line up.",
                "Dropout is applied to each sub-layer's output before it's added to the input and normalized "
                "(P_drop = 0.1).",
            ],
            "focus": [0, 2],
            "code": "DecoderLayer uses norm1, norm2 and norm3 for its three sub-layers.",
        },
        "enc_ffn": {
            "refs": [S33],
            "eq": EQ_FFN,
            "points": [
                "Each layer has a fully connected feed-forward network applied to every position separately "
                "and identically.",
                "It's two linear transformations with a ReLU in between.",
                "The weights are shared across positions but differ from layer to layer; it's equivalent to "
                "two convolutions with kernel size 1.",
                "Input and output have d_model = 512 dims; the inner layer has d_ff = 2048.",
            ],
            "focus": [0, 1],
            "code": f"feedForward.py: Linear {m.d_model}→{m.d_ff}, {m.activation}, Dropout, Linear {m.d_ff}→{m.d_model}. "
                    "The extra dropout between the two linears is not described in the paper.",
        },
        "encoder_stack": {
            "refs": [S31, F1],
            "points": [
                "The encoder is a stack of N = 6 identical layers.",
                "Each layer has two sub-layers: multi-head self-attention, then a position-wise feed-forward network.",
                "The encoder's final output is the memory that every decoder layer attends to.",
            ],
            "focus": [0, 2],
            "code": f"encoder.py stacks {m.n_encoder_layers} EncoderLayers. The final LayerNorm after the stack "
                    "isn't in the paper (it's common in implementations such as torch.nn.Transformer).",
        },
        "decoder_stack": {
            "refs": [S31, F1],
            "points": [
                "The decoder is also a stack of N = 6 identical layers.",
                "Besides the two encoder sub-layers, each decoder layer adds a third: multi-head attention over "
                "the encoder's output.",
                "Its self-attention is masked so a position can't see later positions.",
            ],
            "focus": [0, 1],
            "code": f"decoder.py stacks {m.n_decoder_layers} DecoderLayers, followed by an extra final LayerNorm.",
        },
        "linear": {
            "refs": [S34],
            "points": [
                "A learned linear transformation followed by a softmax turns the decoder's output into "
                "next-token probabilities.",
                "The pre-softmax linear layer shares its weight matrix with the embedding layers.",
            ],
            "focus": [0, 1],
            "code": "Transformer.generator maps d_model → tgt_vocab_size and is tied to outputEmbedding.py.",
        },
        "softmax": {
            "refs": [S34, S54],
            "points": [
                "A learned linear transformation followed by a softmax turns the decoder's output into "
                "next-token probabilities.",
                "During training, label smoothing (ε_ls = 0.1) is used: it hurts perplexity but improves "
                "accuracy and BLEU.",
            ],
            "focus": [0],
            "code": f"The model returns logits; softmax lives in the loss or decoding code. "
                    f"config.yml label_smoothing = {t.loss.label_smoothing}.",
        },
        "padding_mask": {
            "refs": [S51],
            "points": [
                "The paper doesn't describe padding masks. They're an implementation detail needed because "
                "sentences in a batch have different lengths.",
                "Training batches group sentence pairs of similar length, which keeps padding small.",
            ],
            "focus": [0],
            "code": "Transformer.make_padding_mask marks pad tokens so attention gives them zero weight.",
        },
        "self_attention": {
            "refs": [S321],
            "eq": EQ_ATTN,
            "points": ATTN_POINTS[:2],
            "focus": [0],
            "code": "selfAttention.py is a single-head version of Eq. 1. The model uses MultiHeadAttention instead.",
        },
    }


def _comparison(m, t) -> list[dict]:
    """Base-model values from the paper vs the values in config.yml."""
    rows = [
        ("N (encoder layers)", 6, m.n_encoder_layers),
        ("N (decoder layers)", 6, m.n_decoder_layers),
        ("d_model", 512, m.d_model),
        ("h (heads)", 8, m.n_heads),
        ("d_k = d_v", 64, m.d_model // m.n_heads),
        ("d_ff", 2048, m.d_ff),
        ("P_drop", 0.1, m.dropout),
        ("activation", "relu", m.activation),
        ("positional encoding", "sinusoidal", m.positional_encoding),
        ("label smoothing ε_ls", 0.1, t.loss.label_smoothing),
        ("warmup_steps", 4000, t.scheduler.warmup_steps),
        ("Adam β1, β2", "0.9, 0.98", ", ".join(str(b) for b in t.optimizer.betas)),
        ("Adam ε", "1e-09", f"{t.optimizer.eps:g}"),
        ("LR schedule", "noam (Eq. 3)", t.scheduler.name),
    ]
    return [{"name": n, "paper": str(p), "code": str(c), "same": str(p) == str(c)} for n, p, c in rows]


# ----------------------------------------------------------------------------------
# Where each note lives in the paper: (start phrase, end phrase) used only to locate
# the passage in the user's local PDF for highlighting. Keys are the opening words of
# the note text in `points`. Notes the paper doesn't cover have no anchor.
# ----------------------------------------------------------------------------------
ANCHORS = {
    "The model is an encoder": ("Here, the encoder maps an input sequence", "one element at a time"),
    "Generation is auto-regressive": ("At each step the model is auto-regressive", "when generating the next"),
    "Both halves are built": ("The Transformer follows this overall architecture", "of Figure 1, respectively"),
    "Learned embeddings map": ("we use learned embeddings to convert the input", "vectors of dimension"),
    "The paper shares one weight": ("In our model, we share the same weight matrix", "similar to [30]"),
    "Inside the embedding layers": ("In the embedding layers, we multiply those weights by", None),
    "The decoder's input is the output": ("This masking, combined with fact that the output embeddings",
                                          "known outputs at positions less than i"),
    "With no recurrence or convolution": ("Since our model contains no recurrence and no convolution",
                                          "of the tokens in the sequence"),
    "Positional encodings of size": ("To this end, we add", "can be summed"),
    "Each dimension is a sinusoid": ("That is, each dimension of the positional encoding", "geometric progression from"),
    "Sinusoids were chosen": ("We chose this function because we hypothesized", "represented as a linear function of"),
    "Learned positional embeddings gave": ("We also experimented with using learned positional embeddings",
                                           "encountered during training"),
    "Dropout (P_drop = 0.1) is applied to the sum": ("In addition, we apply dropout to the sums of the embeddings",
                                                     "encoder and decoder stacks"),
    "Scaled dot-product attention takes": ("We compute the dot products of the", "obtain the weights on the"),
    "The √d_k scaling matters": ("We suspect that for large values of", "we scale the dot products by"),
    # Spans a page break with a footnote in between, so it's two passages.
    "Multi-head attention projects": [("we found it beneficial to linearly project the queries", "in parallel, yielding"),
                                      ("output values. These are concatenated", "resulting in the final values")],
    "The base model uses h = 8": ("In this work we employ h = 8 parallel attention layers",
                                  "single-head attention with full dimensionality"),
    "In encoder self-attention": ("The encoder contains self-attention layers", "previous layer of the encoder"),
    "Decoder self-attention lets": ("Similarly, self-attention layers in the decoder allow",
                                    "up to and including that position"),
    "To keep the model auto-regressive": ("We need to prevent leftward information flow",
                                          "correspond to illegal connections"),
    "The decoder's self-attention is modified": ("We also modify the self-attention sub-layer in the decoder stack",
                                                 "attending to subsequent positions"),
    "Together with the one-position shift": ("This masking, combined with fact", "positions less than i"),
    "The masking is done inside": ("We implement this inside of scaled dot-product attention",
                                   "correspond to illegal connections"),
    "In encoder–decoder attention": ('In "encoder-decoder attention" layers', "come from the output of the encoder"),
    "This lets every decoder position": ("This allows every position in the decoder", "such as [38, 2, 9]"),
    "Each sub-layer has a residual": ("We employ a residual connection [11]", "implemented by the sub-layer itself"),
    "All sub-layers and embedding layers": ("To facilitate these residual connections", "produce outputs of dimension"),
    "Dropout is applied to each sub-layer": ("We apply dropout [33] to the output of each sub-layer",
                                             "sub-layer input and normalized"),
    "Like the encoder, every decoder": ("Similar to the encoder, we employ residual connections",
                                        "followed by layer normalization"),
    "Each layer has a fully connected": ("In addition to attention sub-layers", "separately and identically"),
    "It's two linear transformations": ("This consists of two linear transformations", "activation in between"),
    "The weights are shared across": ("While the linear transformations are the same", "convolutions with kernel size 1"),
    "Input and output have d_model": ("The dimensionality of input and output is", "inner-layer has dimensionality"),
    "The encoder is a stack": ("Encoder: The encoder is composed of a stack", "identical layers"),
    "Each layer has two sub-layers": ("Each layer has two sub-layers", "fully connected feed-forward network"),
    "The encoder's final output": ("the memory keys and values come from the output of the encoder", None),
    "The decoder is also a stack": ("Decoder: The decoder is also composed of a stack", "identical layers"),
    "Besides the two encoder sub-layers": ("In addition to the two sub-layers in each encoder layer",
                                           "output of the encoder stack"),
    "Its self-attention is masked": ("We also modify the self-attention sub-layer", "attending to subsequent positions"),
    "A learned linear transformation": ("We also use the usual learned linear transformation", "next-token probabilities"),
    "The pre-softmax linear layer": ("In our model, we share the same weight matrix", "similar to [30]"),
    "During training, label smoothing": ("Label Smoothing During training", "accuracy and BLEU score"),
    "Training batches group": ("Sentence pairs were batched together by approximate sequence length", None),
}

# Phrases that locate each section heading / figure caption (for the "open section" page).
SECTION_ANCHORS = {
    "S3": "3 Model Architecture", "S3.F1": "Figure 1: The Transformer", "S3.SS1": "3.1 Encoder and Decoder Stacks",
    "S3.SS2.SSS1": "3.2.1 Scaled Dot-Product Attention", "S3.SS2.SSS2": "3.2.2 Multi-Head Attention",
    "S3.SS2.SSS3": "3.2.3 Applications of Attention", "S3.F2": "Figure 2: (left) Scaled Dot-Product",
    "S3.SS3": "3.3 Position-wise Feed-Forward", "S3.SS4": "3.4 Embeddings and Softmax",
    "S3.SS5": "Since our model contains no recurrence and no convolution", "S5.SS1": "5.1 Training Data and Batching",
    "S5.SS3": "5.3 Optimizer", "S5.SS4": "5.4 Regularization", "S6.SS2": "6.2 Model Variations",
}


def _attach_pdf(entries: dict, pdf) -> dict | None:
    """Add page numbers and highlight boxes from the local PDF; return page image info."""
    if pdf is None:
        for e in entries.values():
            e["points"] = [{"text": t, "hl": None} for t in e["points"]] \
                if e["points"] and isinstance(e["points"][0], str) else e["points"]
        return None

    used_pages: set[int] = set()
    missing = []
    for e in entries.values():
        if e["points"] and isinstance(e["points"][0], dict):
            continue  # shared entry already processed
        pts = []
        for text in e["points"]:
            anchor = next((v for k, v in ANCHORS.items() if text.startswith(k)), None)
            spans = anchor if isinstance(anchor, list) else [anchor] if anchor else []
            hl = []
            for span in spans:
                found = pdf.highlight(*span)
                if found is None:
                    missing.append(span[0])
                else:
                    for f in found:  # merge rects that land on the same page
                        same = next((h for h in hl if h["page"] == f["page"]), None)
                        if same:
                            same["rects"] += f["rects"]
                        else:
                            hl.append(f)
            hl = hl or None
            for h in hl or []:
                used_pages.add(h["page"])
            pts.append({"text": text, "hl": hl})
        e["points"] = pts
        for r in e["refs"]:
            anchor_id = r["url"].rsplit("#", 1)[-1]
            page = pdf.page_of(SECTION_ANCHORS[anchor_id]) if anchor_id in SECTION_ANCHORS else None
            r["page"] = page
            if page:
                used_pages.add(page)
    if missing:
        raise SystemExit("[build_ui] Paper passages not found in the PDF:\n  " + "\n  ".join(missing))
    sizes = pdf.page_sizes()
    return {"file": pdf.pdf.name, "npages": len(sizes), "sizes": sizes}


def paper_data(settings) -> dict:
    m, t = settings.model, settings.training
    entries = _entries(m, t)
    # Components that reuse another entry.
    entries["dec_ffn"] = entries["enc_ffn"]
    entries["encoder_layer"] = entries["encoder_stack"]
    entries["decoder_layer"] = entries["decoder_stack"]
    entries["main"] = entries["transformer"]
    entries["config"] = {
        "refs": [S3, S54, ("5.3", "Optimizer", "S5.SS3")],
        "points": ["The table compares the paper's base model with the values in config.yml."],
        "focus": [],
        "code": "Rows that differ from the paper are highlighted.",
        "table": _comparison(m, t),
    }
    for e in entries.values():
        e["refs"] = [{"sec": s, "title": ti, "url": f"{PAPER_URL}#{a}"} for s, ti, a in e["refs"]] \
            if e["refs"] and isinstance(e["refs"][0], tuple) else e["refs"]

    from paper_pdf import open_paper  # noqa: E402

    local = _attach_pdf(entries, open_paper(Path(__file__).resolve().parent, "attention"))
    return {
        "title": PAPER_TITLE, "url": PAPER_URL, "pdf": PAPER_PDF, "entries": entries,
        "citation": CITATION, "notice": GOOGLE_NOTICE, "disclaimer": DISCLAIMER,
        "local": local, "local_notice": LOCAL_NOTICE,
    }
