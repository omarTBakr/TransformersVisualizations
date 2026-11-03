/*
 * Vision Transformer diagram for the shared page (shared/ui/template.html): an original drawing
 * after the idea of Figure 1 of "An Image Is Worth 16x16 Words" (patches → linear projection →
 * [class] + position embedding → L × pre-norm encoder → MLP head → class), drawn bottom to top.
 *
 *   blocks  { key, id (component), x, y, w, h, lines, small?, pill?, mono?, dashed? }
 *   arrows  { d (SVG path), from: [block keys], to: [block keys], dot?, dashed? }: lit while those blocks run
 *   back / mid  extra SVG drawn under the arrows / between the arrows and the blocks
 *
 * The extra Linear after the projection is not in the paper: its block is dashed and tagged.
 */
window.Model = {
  chips: cfg => [
    ["image", `${cfg.image_size}×${cfg.image_size}×${cfg.in_channels}`], ["patch P", cfg.patch_size],
    ["patches N", cfg.n_patches], ["D", cfg.d_model], ["heads", cfg.n_heads], ["layers L", cfg.n_encoder_layers],
    ["MLP size", cfg.d_ff], ["activation", cfg.activation], ["position", cfg.positional_encoding],
    ["classes K", cfg.num_classes],
  ],
  extras: ["vit", "image_embedding", "config", "main"],
  // Diagram keys that aren't a step themselves → the step to jump to.
  alias: { encoder_layer: "enc_norm1" },
  diagram({ cfg }) {
    const cx = 380, W = 250, X = cx - W / 2, sW = 46, sX = cx - sW / 2;
    const blocks = [
      { key: "patch_embed", id: "patch_embedding", x: X, y: 780, w: W, h: 52, lines: ["Linear Projection of", "Flattened Patches"] },
      { key: "extra_linear", id: "extra_linear", x: X, y: 722, w: W, h: 34, lines: ["Extra Linear (D → D)"], small: true, dashed: true },
      { key: "cls_token", id: "cls_token", x: 60, y: 668, w: 150, h: 36, lines: ["[class] token", "x_class"], small: true },
      { key: "pe", id: "positional_encoding", x: 550, y: 618, w: 150, h: 36, lines: ["Position", "Embedding"], small: true },
      { key: "enc_norm1", id: "enc_norm", x: X, y: 558, w: W, h: 28, lines: ["Norm"], small: true },
      { key: "enc_self_attn", id: "enc_self_attn", x: X, y: 492, w: W, h: 44, lines: ["Multi-Head Attention"] },
      { key: "enc_res1", id: "enc_residual", x: sX, y: 452, w: sW, h: 24, lines: ["+"], pill: true },
      { key: "enc_norm2", id: "enc_norm", x: X, y: 400, w: W, h: 28, lines: ["Norm"], small: true },
      { key: "enc_ffn", id: "enc_ffn", x: X, y: 334, w: W, h: 44, lines: ["MLP"] },
      { key: "enc_res2", id: "enc_residual", x: sX, y: 294, w: sW, h: 24, lines: ["+"], pill: true },
      { key: "encoder_layer", id: "encoder_layer", x: 40, y: 420, w: 150, h: 26, lines: [`Encoder ×${cfg.n_encoder_layers}`], small: true, pill: true },
      { key: "encoder_stack", id: "encoder_stack", x: X, y: 222, w: W, h: 28, lines: ["Norm (final)"], small: true },
      { key: "cls_select", id: "cls_select", x: X + 40, y: 168, w: W - 80, h: 28, lines: ["take [CLS]: x[:, 0]"], small: true, mono: true },
      { key: "head", id: "head", x: X, y: 104, w: W, h: 40, lines: ["MLP Head"] },
      { key: "softmax", id: "softmax", x: X + 40, y: 44, w: W - 80, h: 34, lines: ["Class (softmax)"], small: true },
    ];
    const skip = (y1, y2) => `M${cx} ${y1} H${cx + 170} V${y2} H${cx + sW / 2 + 2}`;
    const arrows = [
      { d: `M${cx} 868 V832`, from: [], to: ["patch_embed"] },
      { d: `M${cx} 780 V756`, from: ["patch_embed"], to: ["extra_linear"] },
      { d: `M${cx} 722 V697`, from: ["extra_linear"], to: ["cls_token"] },
      { d: `M210 686 H${cx - 13}`, from: ["cls_token"], to: ["cls_token"] },
      { d: `M${cx} 675 V647`, from: ["cls_token"], to: ["pe"] },
      { d: `M550 636 H${cx + 13}`, from: [], to: ["pe"] },
      { d: `M${cx} 625 V586`, from: ["pe"], to: ["enc_norm1"] },
      { d: `M${cx} 558 V536`, from: ["enc_norm1"], to: ["enc_self_attn"] },
      { d: `M${cx} 492 V476`, from: ["enc_self_attn"], to: ["enc_res1"] },
      { d: skip(604, 464), from: ["pe", "enc_res2"], to: ["enc_res1"], dot: [cx, 604] },
      { d: `M${cx} 452 V428`, from: ["enc_res1"], to: ["enc_norm2"] },
      { d: `M${cx} 400 V378`, from: ["enc_norm2"], to: ["enc_ffn"] },
      { d: `M${cx} 334 V318`, from: ["enc_ffn"], to: ["enc_res2"] },
      { d: skip(440, 306), from: ["enc_res1"], to: ["enc_res2"], dot: [cx, 440] },
      { d: `M${cx} 294 V250`, from: ["enc_res2"], to: ["encoder_stack"] },
      { d: `M${cx} 222 V196`, from: ["encoder_stack"], to: ["cls_select"] },
      { d: `M${cx} 168 V144`, from: ["cls_select"], to: ["head"] },
      { d: `M${cx} 104 V78`, from: ["head"], to: ["softmax"] },
      { d: `M${cx} 44 V26`, from: ["softmax"], to: [] },
    ];
    // The input image as a strip of patches (decoration), with [CLS] marked as token 0.
    const n = Math.min(9, cfg.n_patches), pw = 20, px0 = cx - (n * (pw + 4)) / 2;
    const strip = Array.from({ length: n }, (_, i) =>
      `<rect class="pstrip" x="${px0 + i * (pw + 4)}" y="872" width="${pw}" height="${pw}" rx="2"/>
       <text class="lbl" x="${px0 + i * (pw + 4) + pw / 2}" y="886" text-anchor="middle" font-size="8">${i + 1}</text>`).join("");
    const back = `
      <rect class="bracket" data-which="enc" x="${cx - 160}" y="282" width="${W + 160}" height="318" rx="14"/>
      <text class="badge-t" data-which="enc" x="${cx - 150}" y="298"></text>
      <text class="lbl" x="${cx - 150}" y="594">Transformer Encoder</text>
      ${strip}
      <text class="lbl" x="${cx}" y="912" text-anchor="middle">Image → ${cfg.n_patches} patches of ${cfg.patch_size}×${cfg.patch_size}×${cfg.in_channels}${cfg.n_patches > n ? " (first 9 drawn)" : ""}</text>
      <text class="np-tag" x="${X + W + 8}" y="743">not in the paper</text>
      <text class="lbl" x="${cx - 18}" y="712" text-anchor="end">prepend</text>
      <text class="lbl-strong" x="${cx}" y="18" text-anchor="middle">Class probabilities</text>
    `;
    const mid = `<g class="plus" data-key="cls_token"><circle cx="${cx}" cy="686" r="11"/><text x="${cx}" y="690.5" text-anchor="middle" font-size="11">⧺</text></g>
      <g class="plus" data-key="pe"><circle cx="${cx}" cy="636" r="11"/><text x="${cx}" y="641" text-anchor="middle">+</text></g>`;
    return { box: [0, 0, 760, 925], aria: "Vision Transformer diagram", blocks, arrows, back, mid };
  },
};
