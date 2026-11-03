/*
 * Encoder–decoder diagram for the shared page (shared/ui/template.html): the blocks of
 * Figure 1 of "Attention Is All You Need", the arrows between them, and the header chips.
 *
 *   blocks  { key, id (component), x, y, w, h, lines, small?, pill?, mono? }
 *   arrows  { d (SVG path), from: [block keys], to: [block keys], dot? }: lit while those blocks run
 *   back / mid  extra SVG drawn under the arrows / between the arrows and the blocks
 *   brackets and badges with data-which="enc" | "dec" light up while a layer of that stack runs
 */
window.Model = {
  chips: cfg => [
    ["d_model", cfg.d_model], ["heads", cfg.n_heads], ["encoder layers", cfg.n_encoder_layers],
    ["decoder layers", cfg.n_decoder_layers], ["d_ff", cfg.d_ff], ["dropout", cfg.dropout],
    ["activation", cfg.activation], ["positional", cfg.positional_encoding], ["max_seq_len", cfg.max_seq_len],
    ["vocab src/tgt", `${cfg.src_vocab_size} / ${cfg.tgt_vocab_size}`], ["tied embeddings", cfg.share_embeddings ? "yes" : "no"],
  ],
  extras: ["transformer", "padding_mask", "config", "self_attention", "main"],
  // Diagram keys that aren't a step themselves → the step to jump to.
  alias: { encoder_layer: "enc_self_attn", decoder_layer: "dec_masked_attn" },
  diagram({ cfg }) {
  const EX = 120, DX = 480, W = 200, ecx = EX + W / 2, dcx = DX + W / 2;
  const blocks = [
    { key: "input_embedding", id: "input_embedding", x: EX, y: 840, w: W, h: 40, lines: ["Input Embedding"] },
    { key: "pe_enc", id: "positional_encoding", x: 4, y: 790, w: 86, h: 34, lines: ["Positional", "Encoding"], small: true },
    { key: "enc_self_attn", id: "enc_self_attn", x: EX, y: 700, w: W, h: 45, lines: ["Multi-Head Attention"] },
    { key: "enc_norm1", id: "enc_add_norm", x: EX, y: 655, w: W, h: 30, lines: ["Add & Norm"], small: true },
    { key: "enc_ffn", id: "enc_ffn", x: EX, y: 585, w: W, h: 45, lines: ["Feed Forward"] },
    { key: "enc_norm2", id: "enc_add_norm", x: EX, y: 540, w: W, h: 30, lines: ["Add & Norm"], small: true },
    { key: "encoder_stack", id: "encoder_stack", x: 4, y: 606, w: 86, h: 26, lines: [`Encoder ×${cfg.n_encoder_layers}`], small: true, pill: true },
    { key: "encoder_layer", id: "encoder_layer", x: 4, y: 640, w: 86, h: 26, lines: ["EncoderLayer"], small: true, pill: true, mono: true },
    { key: "output_embedding", id: "output_embedding", x: DX, y: 840, w: W, h: 40, lines: ["Output Embedding"] },
    { key: "pe_dec", id: "positional_encoding", x: 728, y: 790, w: 86, h: 34, lines: ["Positional", "Encoding"], small: true },
    { key: "dec_masked_attn", id: "dec_masked_attn", x: DX, y: 700, w: W, h: 45, lines: ["Masked Multi-Head Attention"] },
    { key: "dec_norm1", id: "dec_add_norm", x: DX, y: 655, w: W, h: 30, lines: ["Add & Norm"], small: true },
    { key: "dec_cross_attn", id: "dec_cross_attn", x: DX, y: 585, w: W, h: 45, lines: ["Multi-Head Attention"] },
    { key: "dec_norm2", id: "dec_add_norm", x: DX, y: 540, w: W, h: 30, lines: ["Add & Norm"], small: true },
    { key: "dec_ffn", id: "dec_ffn", x: DX, y: 470, w: W, h: 45, lines: ["Feed Forward"] },
    { key: "dec_norm3", id: "dec_add_norm", x: DX, y: 425, w: W, h: 30, lines: ["Add & Norm"], small: true },
    { key: "decoder_stack", id: "decoder_stack", x: 722, y: 560, w: 94, h: 26, lines: [`Decoder ×${cfg.n_decoder_layers}`], small: true, pill: true },
    { key: "decoder_layer", id: "decoder_layer", x: 722, y: 594, w: 94, h: 26, lines: ["DecoderLayer"], small: true, pill: true, mono: true },
    { key: "linear", id: "linear", x: DX, y: 330, w: W, h: 40, lines: ["Linear"] },
    { key: "softmax", id: "softmax", x: DX, y: 260, w: W, h: 40, lines: ["Softmax"] },
  ];

  // Arrows: `from` / `to` list the block keys whose activity lights the arrow up.
  const res = (cx, xr, y1, y2) => `M${cx} ${y1} H${xr} V${y2} H${xr - 15}`;
  const arrows = [
    { d: `M${ecx} 895 V880`, from: [], to: ["input_embedding"] },
    { d: `M${ecx} 840 V818`, from: ["input_embedding"], to: ["pe_enc"] },
    { d: `M90 807 H${ecx - 12}`, from: [], to: ["pe_enc"] },
    { d: `M${ecx} 795 V745`, from: ["pe_enc"], to: ["enc_self_attn"] },
    { d: res(ecx, 335, 765, 670), from: ["pe_enc"], to: ["enc_norm1"], dot: [ecx, 765] },
    { d: `M${ecx} 700 V685`, from: ["enc_self_attn"], to: ["enc_norm1"] },
    { d: `M${ecx} 655 V630`, from: ["enc_norm1"], to: ["enc_ffn"] },
    { d: res(ecx, 335, 645, 555), from: ["enc_norm1"], to: ["enc_norm2"], dot: [ecx, 645] },
    { d: `M${ecx} 585 V570`, from: ["enc_ffn"], to: ["enc_norm2"] },
    { d: `M${ecx} 540 V500 H420 V645 H${DX + 40} V630`, from: ["encoder_stack"], to: ["encoder_stack", "dec_cross_attn"] },
    { d: `M${dcx} 895 V880`, from: [], to: ["output_embedding"] },
    { d: `M${dcx} 840 V818`, from: ["output_embedding"], to: ["pe_dec"] },
    { d: `M728 807 H${dcx + 12}`, from: [], to: ["pe_dec"] },
    { d: `M${dcx} 795 V745`, from: ["pe_dec"], to: ["dec_masked_attn"] },
    { d: res(dcx, 695, 765, 670), from: ["pe_dec"], to: ["dec_norm1"], dot: [dcx, 765] },
    { d: `M${dcx} 700 V685`, from: ["dec_masked_attn"], to: ["dec_norm1"] },
    { d: `M${dcx} 655 V630`, from: ["dec_norm1"], to: ["dec_cross_attn"] },
    { d: res(dcx, 695, 645, 555), from: ["dec_norm1"], to: ["dec_norm2"], dot: [dcx, 645] },
    { d: `M${dcx} 585 V570`, from: ["dec_cross_attn"], to: ["dec_norm2"] },
    { d: `M${dcx} 540 V515`, from: ["dec_norm2"], to: ["dec_ffn"] },
    { d: res(dcx, 695, 530, 440), from: ["dec_norm2"], to: ["dec_norm3"], dot: [dcx, 530] },
    { d: `M${dcx} 470 V455`, from: ["dec_ffn"], to: ["dec_norm3"] },
    { d: `M${dcx} 425 V370`, from: ["decoder_stack"], to: ["decoder_stack", "linear"] },
    { d: `M${dcx} 330 V300`, from: ["linear"], to: ["softmax"] },
    { d: `M${dcx} 260 V244`, from: ["softmax"], to: [] },
  ];
    const back = `
    <rect class="bracket" data-which="enc" x="${EX - 25}" y="520" width="${W + 50}" height="250" rx="14"/>
    <rect class="bracket" data-which="dec" x="${DX - 25}" y="405" width="${W + 50}" height="365" rx="14"/>
    <text class="badge-t" data-which="enc" x="${EX - 17}" y="534"></text>
    <text class="badge-t" data-which="dec" x="${DX - 17}" y="419"></text>
    <text class="lbl" x="${ecx}" y="910" text-anchor="middle">Inputs</text>
    <text class="lbl" x="${dcx}" y="910" text-anchor="middle">Outputs (shifted right)</text>
    <text class="lbl-strong" x="${dcx}" y="236" text-anchor="middle">Output Probabilities</text>
    <text class="lbl" x="318" y="493" text-anchor="middle">memory (K, V)</text>
    `;
    const mid = `<g class="plus" data-key="pe_enc"><circle cx="${ecx}" cy="807" r="11"/><text x="${ecx}" y="812" text-anchor="middle">+</text></g>
          <g class="plus" data-key="pe_dec"><circle cx="${dcx}" cy="807" r="11"/><text x="${dcx}" y="812" text-anchor="middle">+</text></g>`;
    return { box: [0, 222, 820, 700], aria: "Transformer encoder-decoder diagram", blocks, arrows, back, mid };
  },
};
