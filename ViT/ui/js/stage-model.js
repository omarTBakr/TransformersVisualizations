/*
 * Vision Transformer steps for the shared operation stage (shared/ui/js/stage-core.js):
 * image → patches → projection (Conv2d as a matmul) → extra Linear → [CLS] → position embedding,
 * the pre-norm residual adds, the [CLS] selection, the head and the class softmax, plus the
 * breadcrumb. LayerNorm, attention and the MLP come from the core.
 */
window.StageModel = function (api) {
  "use strict";
  const { T, cfg, B, d, esc, fmt, sample, subIdx, byLabel, words, nf, cubeOf, subPart, shapeOnly, eqTerm,
          mat, matOf, transpose, G, rowMaxFlag, argmaxAbs, txt, t3, ts3, n3, s3, matmul, linearStage,
          softmaxCell, pickOf, checkTxt, checkTex } = api;
  const DET = () => api.DETAIL;
  const P = cfg.patch_size, C = cfg.in_channels, N = cfg.n_patches, gridN = cfg.image_size / P;
  const PPC = P * P * C;
  const pLab = i => `p${i + 1}`;
  const imgLab = i => `image ${i + 1}`;
  const className = k => B.tgt_vocab[k] ?? `class ${k}`;
  const cutNote = sl => (sl && sl.T ? ` (first ${sl.v[0].length} of ${sl.T} tokens)` : "");

  // ---------------------------------------------------------------- the image, drawn as pixels
  // ex = patch_embed.extra: { image: [B][C][h][w] thumbnail, stride, size, patch, channels, grid }
  function imagePart(ex, { grid = false, hi = null, values = false, title = "" } = {}) {
    const im = ex.image[sample()], h = im[0].length, w = im[0][0].length;
    const cs = Math.max(4, Math.min(30, Math.floor(260 / w)));      // cell size in px
    const pp = ex.patch / ex.stride;                                 // patch size in thumbnail pixels
    const top = title ? 18 : 4, left = 4;
    let body = title ? `<text class="ptitle" x="${left}" y="12">${esc(title)}</text>` : "";
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const px = im.map(ch => ch[y][x]), g = px.length === 1;
      const [r, gg, b] = (g ? [px[0], px[0], px[0]] : px).map(v => Math.round(255 * Math.max(0, Math.min(1, v))));
      body += `<rect x="${left + x * cs}" y="${top + y * cs}" width="${cs}" height="${cs}" style="fill:rgb(${r},${gg},${b})"><title>(${y}, ${x}): ${px.map(nf).join(", ")}</title></rect>`;
      if (values && g && cs >= 22) body += `<text class="pxv${px[0] > 0.5 ? " dark" : ""}" x="${left + x * cs + cs / 2}" y="${top + y * cs + cs / 2 + 3.5}" text-anchor="middle">${n3(px[0]).replace(/0+$/, "").replace(/\.$/, "")}</text>`;
    }
    if (grid) for (let k = 0; k < ex.grid * ex.grid; k++) {
      const pr = Math.floor(k / ex.grid), pc = k % ex.grid, many = ex.grid > 6;
      body += `<rect class="pgrid${many ? " thin" : ""}" x="${left + pc * pp * cs}" y="${top + pr * pp * cs}" width="${pp * cs}" height="${pp * cs}"/>`;
      // ViT-Base (14 × 14 patches): number the first row only, plus the highlighted patch.
      if (!many || pr === 0 && pc % 2 === 0 || k === hi)
        body += `<text class="plab" x="${left + pc * pp * cs + 2}" y="${top + pr * pp * cs + 9}">${pLab(k)}</text>`;
    }
    if (hi != null) {
      const pr = Math.floor(hi / ex.grid), pc = hi % ex.grid;
      body += `<rect class="hicell big" x="${left + pc * pp * cs}" y="${top + pr * pp * cs}" width="${pp * cs}" height="${pp * cs}" rx="2"/>`;
    }
    return { w: left + w * cs + 6, h: top + h * cs + 6, body };
  }

  // Feature names of a flattened patch: pixel (row, col) of channel c, in PyTorch's (c, row, col) order.
  const featLab = k => { const c = Math.floor(k / (P * P)), r = Math.floor((k % (P * P)) / P), q = k % P;
    return C > 1 ? `${"RGB"[c] || "c" + c}${r},${q}` : `px${r},${q}`; };

  // ---------------------------------------------------------------- patch embedding
  function patchStages(st) {
    const ex = st.extra, conv = subIdx(st, byLabel(/^Conv2d/)), fl = subIdx(st, byLabel(/^Flatten/)), tr = subIdx(st, byLabel(/^Swap axes/));
    const S = st.subs, ops = S[conv] && S[conv].ops;
    const out = () => { const m = mat(S[tr]); if (m) m.rowLab = pLab; return m; };
    const patchRows = () => (ops ? matOf(ops.x[sample()], "patches", "pixels") : null);
    const stages = [
      { title: "The input image", term: "x", sub: null,
        how: `A ${cfg.image_size}×${cfg.image_size} image with C = ${C} channel${C > 1 ? "s" : ""}: H·W·C = ${cfg.image_size ** 2 * C} numbers. Pixel values here are synthetic, in [0, 1].`,
        parts: () => [imagePart(ex, { values: true, title: `image ${sample() + 1}${B.true[sample()] != null ? `: ${className(B.true[sample()])}` : ""}` }),
                      T.note([`x ∈ ℝ^(H×W×C) = ℝ^(${cfg.image_size}×${cfg.image_size}×${C})`],
                        { w: 200, tex: [`x \\in \\mathbb{R}^{H\\times W\\times C} = \\mathbb{R}^{${cfg.image_size}\\times ${cfg.image_size}\\times ${C}}`] })] },
      { title: `Cut into N = ${N} patches of ${P}×${P}`, term: "xp", sub: conv,
        how: `N = HW / P² = ${cfg.image_size ** 2} / ${P * P} = ${N}. Patches are numbered in raster order (row by row); this order is the sequence order.`,
        parts: () => [imagePart(ex, { grid: true, values: true, title: "patches" }),
                      T.note([`N = HW / P² = ${N}`, `each patch: P²·C = ${P}·${P}·${C} = ${PPC} values`],
                        { w: 220, tex: [`N = \\frac{HW}{P^2} = \\frac{${cfg.image_size}\\cdot ${cfg.image_size}}{${P}^2} = ${N}`,
                                        `P^2\\cdot C = ${P}^2\\cdot ${C} = ${PPC}\\ ${txt("values per patch")}`] })] },
    ];
    const nShow = DET() ? N : Math.min(2, N);
    for (let k = 0; k < nShow; k++) stages.push({
      title: `Flatten patch ${pLab(k)}`, term: "xp", sub: conv,
      how: `The ${P}×${P}×${C} pixels of ${pLab(k)} become one row x_p of ${PPC} numbers (channel, then row, then column).`,
      parts: () => {
        const xr = patchRows();
        if (!xr) return [imagePart(ex, { grid: true, hi: k, title: pLab(k) }), T.op("flatten"), T.note([`x_p${k + 1} ∈ ℝ^${PPC}`], { w: 120 })];
        const row = { m: [xr.m[k]], rowLab: () => pLab(k), colLab: featLab, kind: "float" };
        return [imagePart(ex, { grid: true, hi: k, values: true, title: pLab(k) }), T.op("flatten", `${PPC} values`), G(row, { title: `x_p (row ${k + 1})` })];
      } });
    if (ops) stages.push({ title: "Project every patch: x_p · E + b", term: "e", sub: conv, pick: true,
      how: `All N patches at once: (N × ${PPC}) · (${PPC} × ${d}) = (N × ${d}). The Conv2d kernel, reshaped to (${d}, ${PPC}) and transposed, is E. Click a result cell to see its dot product.`,
      parts: () => {
        const A = patchRows(); A.colLab = featLab;
        const E = matOf(transpose(ops.W), "pixels", "features"); E.rowLab = featLab;
        return matmul({ A, Bm: E, Cm: out(), bias: ops.b, tA: "x_p", tB: "E", tC: "z" }).parts();
      } });
    else stages.push({ title: "Project every patch: x_p · E + b", term: "e", sub: conv,
      how: `A Conv2d with kernel = stride = ${P} computes x_p · E for all ${N} patches in one call: ${PPC} inputs → D = ${d} per patch.`,
      parts: () => [imagePart(ex, { grid: true, title: "patches" }), T.op("· E", `${PPC} → ${d}`), subPart(st, tr, { single: true, title: "patch embeddings", xLabels: pLab })] });
    stages.push({ title: "Grid → sequence: flatten + transpose", term: "seq", sub: tr,
      how: `The conv returns a (B, D, ${gridN}, ${gridN}) grid. flatten(2) joins the grid axes in raster order → (B, D, N), and transpose(1, 2) → (B, N, D): one row per patch.`,
      parts: () => {
        if (!DET()) return [T.note([`conv: ${fmt(S[conv].shape)}`, `flatten(2): ${fmt(S[fl].shape)}`, `transpose(1, 2): ${fmt(S[tr].shape)}`], { w: 200 }),
                            subPart(st, tr, { single: true, title: "(B, N, D)", xLabels: pLab })];
        const f = mat(S[fl]); f.colLab = pLab;
        return [G(f, { title: "after flatten(2): D × N" }), T.op("transpose"), G(out(), { title: "after transpose: N × D" })];
      } });
    return {
      title: "Patch embedding (Eq. 1)",
      eq: `z<sub>0</sub> = [x<sub>class</sub>; ${eqTerm("xp", "x<sub>p</sub><sup>1</sup>")}${eqTerm("e", "E")}; …; x<sub>p</sub><sup>N</sup>E] + E<sub>pos</sub>
           &nbsp;·&nbsp; ${eqTerm("x", `x ∈ ℝ<sup>H×W×C</sup>`)} → ${eqTerm("seq", `ℝ<sup>N×D</sup>`)}`,
      stages,
    };
  }

  // ---------------------------------------------------------------- extra Linear (not in the paper)
  function extraLinearStages(st) {
    const li = subIdx(st, byLabel(/^Linear linear_embedding/));
    const s = st.subs[li];
    const compose = T.note([`(x_p·E + b)·Wᵀ + b′ = x_p·(E·Wᵀ) + (b·Wᵀ + b′)`, `= x_p·E′ + b″: still one linear map`, `extra parameters: D² + D = ${d * d + d}`], {
      w: 300, tex: ["(x_p E + b)\\,W^{\\top} + b' = x_p\\,(E W^{\\top}) + (b W^{\\top} + b') = x_p E' + b''",
                    `${txt("extra parameters:")}\\ D^2 + D = ${d * d + d}`] });
    const first = s && s.ops
      ? linearStage(st, li, "y = z · Wᵀ + b′ (every patch)", { term: "w", a: "z", b: "Wᵀ", c: "y", rows: "patches" },
                    "A D → D Linear on each patch embedding. Not in the paper.")
      : { title: "y = z · Wᵀ + b′ (every patch)", term: "w", sub: li, how: "A D → D Linear on each patch embedding. Not in the paper.",
          parts: () => [cubeOf(st.in_vals, { single: true, title: "z", xLabels: pLab }), T.op("× Wᵀ + b′", `${d} → ${d}`), cubeOf(st.out_vals, { single: true, title: "y", xLabels: pLab })] };
    return {
      title: "Extra Linear (not in the paper)",
      eq: `${eqTerm("w", "y = (x<sub>p</sub>E + b) W<sup>T</sup> + b′")} &nbsp;=&nbsp; ${eqTerm("c", "x<sub>p</sub> (E W<sup>T</sup>) + b″")}`,
      stages: [first, { title: "Why it adds nothing", term: "c", sub: li,
        how: "Two linear maps in a row compose into one, so the model could learn E′ = E·Wᵀ directly. The layer adds parameters (and an extra matmul) but no expressive power.",
        parts: () => [compose] }],
    };
  }

  // ---------------------------------------------------------------- [class] token
  function clsStages(st) {
    const ex = subIdx(st, byLabel(/^Repeat \[CLS\]/)), cat = subIdx(st, byLabel(/^Prepend \[CLS\]/));
    const S = st.subs;
    const clsRows = () => { const v = S[ex] && S[ex].vals; return v ? { m: v.v, rowLab: imgLab, colLab: i => `f${i}`, kind: "float" } : null; };
    const patches = () => { if (!st.in_vals) return null; const m = matOf(st.in_vals.v[sample()], "patches", "features"); return m; };
    const outM = () => mat(S[cat]);
    return {
      title: "Prepend the [class] token",
      eq: `z<sub>0</sub> = [${eqTerm("cls", "x<sub>class</sub>")}; ${eqTerm("cat", "x<sub>p</sub><sup>1</sup>E; …; x<sub>p</sub><sup>N</sup>E")}] + E<sub>pos</sub>`,
      stages: [
        { title: "One learned vector x_class", term: "cls", sub: ex,
          how: "x_class is a single (1, 1, D) parameter. expand gives every image the same copy (a view: no memory is copied).",
          parts: () => (DET() ? [G(clsRows(), { title: "x_class per image (identical rows)" })]
                               : [subPart(st, ex, { title: "x_class (B, 1, D)" })]) },
        { title: "Put it in front: position 0", term: "cat", sub: cat,
          how: `torch.cat along the sequence axis: (B, 1, D) + (B, ${N}, D) → (B, ${N + 1}, D). Token 0 is [CLS]; tokens 1…${N} are the patches.`,
          parts: () => (DET() ? [G(patches(), { title: "patch embeddings" }), T.op("prepend"), G(outM(), { title: "tokens", hiRow: 0 })]
                               : [cubeOf(st.in_vals, { single: true, title: "patches", xLabels: pLab }), T.op("prepend"),
                                  cubeOf(st.out_vals, { single: true, title: "tokens" + cutNote(st.out_vals), hiCols: new Set([0]) })]) },
      ],
    };
  }

  // ---------------------------------------------------------------- position embedding
  // Sinusoidal values for the recorded dims, from the formula (config.yml: positional_encoding).
  function peSlice(xs) {
    const v = xs.v.map(seq => seq.map((_, pos) => xs.dims.map(dim => {
      const i2 = dim - (dim % 2), ang = pos / Math.pow(10000, i2 / d);
      return dim % 2 ? Math.cos(ang) : Math.sin(ang);
    })));
    return { v, D: xs.D, dims: xs.dims, T: xs.T };
  }
  function peStages(st) {
    const add = subIdx(st, byLabel(/positional encoding/));
    const X = st.in_vals, Y = st.out_vals, Pe = peSlice(X), learned = cfg.positional_encoding === "learned";
    const tokN = X.v[0].length;
    const stages = [{ title: "z₀ = tokens + E_pos", term: "all", sub: add,
      how: `One ${d}-dim vector per position is added to every image (the same table for the whole batch). ${learned ? "" : "These are the fixed sinusoids of the original Transformer; the paper learns E_pos instead."}`,
      parts: () => (DET() ? [G(matOf(X.v[sample()], "seq", "features"), { title: "tokens" }), T.op("+"), G(matOf(Pe.v[sample()], "seq", "features"), { title: "E_pos" }), T.op("="),
                             G(matOf(Y.v[sample()], "seq", "features"), { title: "z₀" })]
                          : [cubeOf(X, { single: true, title: "tokens", cw: 18 }), T.op("+"), cubeOf(Pe, { single: true, title: "E_pos", cw: 18 }), T.op("="),
                             cubeOf(Y, { single: true, title: "z₀" + cutNote(Y), cw: 18 })]) }];
    if (!learned) Array.from({ length: Math.min(DET() ? tokN : 2, 5) }, (_, t) => t).forEach(t => stages.push({
      title: `Position ${t}: ${words("src")[t]}`, term: "pe", sub: add, pick: DET(),
      how: "Row t of E_pos is added to row t, element by element. Even dims use sin, odd dims cos. The position says where the patch was in raster order; nothing tells the model it is a 2-D grid.",
      parts: () => {
        const b = sample();
        if (!DET()) return [cubeOf(X, { single: true, hiCols: new Set([t]), title: "tokens", cw: 18 }), T.op("+"), cubeOf(Pe, { single: true, hiCols: new Set([t]), title: "E_pos", cw: 18 }),
                            T.op("="), cubeOf(Y, { single: true, hiCols: new Set([t]), title: "z₀", cw: 18 })];
        const Xm = matOf(X.v[b], "seq", "features"), Pm = matOf(Pe.v[b], "seq", "features"), Ym = matOf(Y.v[b], "seq", "features");
        const k = Math.min(9, d);
        return [G(Xm, { title: "tokens", hiRow: t, rowArrow: true }), T.op("+"), G(Pm, { title: "E_pos", hiRow: t, rowArrow: true }), T.op("="), G(Ym, { title: "z₀", hiRow: t }),
          T.note(Xm.m[t].slice(0, k).map((x, j) => `f${j}: ${n3(x)} + ${s3(Pm.m[t][j])} = ${n3(Ym.m[t][j])}`),
            { tex: Xm.m[t].slice(0, k).map((x, j) => { const i2 = j - (j % 2);
              return `z_{${t},${j}} = x_{${t},${j}} + \\${j % 2 ? "cos" : "sin"}\\!\\left(\\frac{${t}}{10000^{${i2}/${d}}}\\right) = ${t3(x)} + ${ts3(Pm.m[t][j])} = ${t3(Ym.m[t][j])}`; }) })];
      } }));
    stages.push({ title: "z₀ enters the encoder", term: "y", sub: add,
      how: `The ${tokN === N + 1 ? N + 1 : B.seq_total} tokens ([CLS] + ${N} patches) are the input sequence of the encoder.`,
      parts: () => [cubeOf(Y, { title: "z₀" + cutNote(Y) }), T.op("→"), T.note(["LN → MSA → +", "LN → MLP → +", `× L = ${cfg.n_encoder_layers}`], { w: 140 })] });
    return {
      title: "Position embedding",
      eq: `${eqTerm("y", "z<sub>0</sub>")} = [x<sub>class</sub>; x<sub>p</sub><sup>1</sup>E; …] + ${eqTerm("pe", "E<sub>pos</sub>")}${eqTerm("all", "")}
           &nbsp;·&nbsp; ${learned ? "E<sub>pos</sub> learned (as in the paper)" : "here E<sub>pos</sub>(t, 2i) = sin(t / 10000<sup>2i/D</sup>), E<sub>pos</sub>(t, 2i+1) = cos(…)"}`,
      stages,
    };
  }

  // ---------------------------------------------------------------- pre-norm residual adds
  function residualStages(st) {
    const after = st.key === "enc_res1" ? "MSA" : "MLP", other = st.key === "enc_res1" ? "attn_output" : "ffn_output";
    const res = subIdx(st, byLabel(/^Residual add/));
    const xs = st.in_all && st.in_all.x, ss = st.in_all && st.in_all[other], ys = st.out_vals;
    const eq = st.key === "enc_res1" ? "z′<sub>ℓ</sub> = MSA(LN(z<sub>ℓ−1</sub>)) + z<sub>ℓ−1</sub>" : "z<sub>ℓ</sub> = MLP(LN(z′<sub>ℓ</sub>)) + z′<sub>ℓ</sub>";
    const stages = [{ title: `x + ${after}(LN(x))`, term: "add", sub: res, pick: DET(),
      how: `The block's output is added to the block's input x, not to the normalized copy: the skip path is never normalized. ${DET() ? "Click a cell." : ""}`,
      parts: () => {
        if (!xs || !ss || !ys) return [shapeOnly(st.out_shape, "residual")];
        if (!DET()) return [cubeOf(xs, { single: true, title: "x", cw: 18 }), T.op("+"), cubeOf(ss, { single: true, title: after.toLowerCase() + " output", cw: 18 }),
                            T.op("="), cubeOf(ys, { single: true, title: "x + " + after + cutNote(ys), cw: 18 })];
        const b = sample(), X = matOf(xs.v[b], "seq", "features"), Sm = matOf(ss.v[b], "seq", "features"), Y = matOf(ys.v[b], "seq", "features");
        const { i, j } = pickOf(argmaxAbs(Sm.m));
        const v = X.m[i][j] + Sm.m[i][j];
        return [G(X, { title: "x", hiCell: { i, j } }), T.op("+"), G(Sm, { title: `${after}(LN(x))`, hiCell: { i, j } }), T.op("="),
                G(Y, { title: "sum (click a cell)", hiCell: { i, j }, gridId: "C" }),
                T.note([`${n3(X.m[i][j])} + ${s3(Sm.m[i][j])} = ${n3(v)}   ${checkTxt(v, Y.m[i][j])}`],
                  { tex: [`${st.key === "enc_res1" ? "z'" : "z"}_{${txt(Y.rowLab(i))},\\,${j}} = ${t3(X.m[i][j])} + ${ts3(Sm.m[i][j])} = ${t3(v)}\\quad ${checkTex(v, Y.m[i][j])}`] })];
      } }];
    return { title: `Residual add (after ${after})`, eq: eqTerm("add", eq), stages };
  }

  // ---------------------------------------------------------------- [CLS] selection
  function clsSelectStages(st) {
    const sel = subIdx(st, byLabel(/^Keep token 0/)), S = st.subs;
    return { title: "Keep the [CLS] token", eq: `${eqTerm("sel", "y = LN(z<sub>L</sub>)<sup>0</sup>")} &nbsp;(Eq. 4)`,
      stages: [{ title: "x[:, 0, :]", term: "sel", sub: sel,
        how: `Only token 0 goes to the head. Through ${cfg.n_encoder_layers} layer${cfg.n_encoder_layers > 1 ? "s" : ""} of attention it has gathered information from every patch; the patch tokens themselves are dropped here.`,
        parts: () => {
          if (!DET()) return [cubeOf(st.in_vals, { single: true, title: "y = LN(z_L)" + cutNote(st.in_vals), hiCols: new Set([0]) }), T.op("x[:, 0]"),
                              S[sel] && S[sel].vals ? T.fromSlice(S[sel].vals, { title: "[CLS] per image" }) : shapeOnly(st.out_shape, "[CLS]")];
          const v = S[sel].vals;
          return [G(matOf(st.in_vals.v[sample()], "seq", "features"), { title: `y, image ${sample() + 1}`, hiRow: 0, rowArrow: true }), T.op("x[:, 0]"),
                  G({ m: v.v, rowLab: imgLab, colLab: i => `f${i}`, kind: "float" }, { title: "[CLS], every image", hiRow: sample() })];
        } }] };
  }

  // ---------------------------------------------------------------- classification head
  function headStages(st) {
    const ex = st.extra || {}, K = cfg.num_classes;
    const stage = st.ops
      ? { title: `logits = y · Wᵀ + b (${d} → ${K} classes)`, term: "l", sub: null, pick: true,
          how: "One score per class for every image. Click a cell to see its dot product.",
          parts: () => {
            const A = { m: st.ops.x, rowLab: imgLab, colLab: i => `f${i}`, names: ["batch", "features"], kind: "float" };
            const Wt = matOf(transpose(st.ops.W), "features", "classes");
            const Cm = { m: ex.out2, rowLab: imgLab, colLab: className, names: ["batch", "classes"], kind: "float" };
            return matmul({ A, Bm: Wt, Cm, bias: st.ops.b, tA: "y", tB: "Wᵀ", tC: "logit" }).parts();
          } }
      : { title: `logits = y · Wᵀ + b (${d} → ${K} classes)`, term: "l", sub: null,
          how: "One Linear layer: the paper's fine-tuning head.",
          parts: () => { const tk = st.topk_logits;
            return [T.note([`y: ${fmt(st.inputs[0].shape)}`, `W: (${K}, ${d})`, `logits: ${fmt(st.out_shape)}`], { w: 160 }),
                    tk ? T.note(["top logits, image 1:", ...tk.ids.slice(0, 3).map((id, i) => `${className(id)}: ${tk.logits[i].toFixed(2)}`)], { w: 190 }) : null].filter(Boolean); } };
    return { title: "Classification head", eq: eqTerm("l", "logits = y W<sup>T</sup> + b, &nbsp; W ∈ ℝ<sup>K×D</sup>"), stages: [stage] };
  }

  // ---------------------------------------------------------------- class softmax (one row per image)
  function softmaxStages(st) {
    const ex = st.extra, S = st.subs, K = cfg.num_classes;
    // ViT-Base has 1000 classes: show the 8 most likely for the selected image.
    const cols = () => {
      const p = ex.out2[sample()], all = p.map((_, k) => k);
      return K <= 12 ? all : all.sort((a, b) => p[b] - p[a]).slice(0, 8);
    };
    const M = (rows, title) => { const c = cols(); return { m: rows.map(r => c.map(k => r[k])), rowLab: imgLab, colLab: j => className(c[j]), kind: "float", title }; };
    const Z = () => M(ex.in2), E = () => M(S[1].vals.v), Pm = () => M(ex.out2);
    const col = v => ({ m: v.map(x => [x]), rowLab: imgLab, colLab: () => "", kind: "float" });
    const sub = K <= 12 ? "" : " (top 8 of " + K + ")";
    return { title: "Softmax over the classes",
      eq: `p<sub>k</sub> = ${eqTerm("e", "exp(z<sub>k</sub> − " + eqTerm("m", "m") + ")")} / ${eqTerm("s", "Σ<sub>j</sub> exp(z<sub>j</sub> − m)")} ${eqTerm("p", "")}`,
      stages: [
        { title: "Row max m (for stability)", term: "m", sub: 0, how: "The largest logit of each image is marked ★ and subtracted before exp, so exp can't overflow.",
          parts: () => [G(Z(), { title: "logits z" + sub, flags: rowMaxFlag(Z().m) }), T.op("max →"), G(col(S[0].vals.v), { title: "m" })] },
        { title: "Exponentiate e = exp(z − m)", term: "e", sub: 1, how: "The row max becomes exp(0) = 1; everything else is smaller.",
          parts: () => [G(Z(), { title: "z" + sub }), T.op("exp(z−m)"), G(E(), { title: "e", flags: rowMaxFlag(E().m) })] },
        { title: "Sum Σ over the classes", term: "s", sub: 2, how: "Add up each row (over all classes, not only the ones shown).",
          parts: () => [G(E(), { title: "e" + sub, rowArrow: true, hiRow: sample() }), T.op("Σ →"), G(col(S[2].vals.v), { title: "Σ" })] },
        { title: "Divide p = e / Σ", term: "p", sub: 3, pick: true,
          how: `Each row now sums to 1; the most likely class is marked ★. The weights are untrained, so p is close to uniform (1 / ${K}). Click a probability.`,
          parts: () => {
            const p = Pm(), z = Z(), { i, j } = pickOf({ i: sample(), j: p.m[sample()].indexOf(Math.max(...p.m[sample()])) });
            const cellN = K <= 12 ? softmaxCell(z.m[i], i, j, p.rowLab(i), p.colLab(j), "p", "z", p.m[i][j]) : null;
            return [G(p, { title: "p (click a cell)" + sub, hiCell: { i, j }, gridId: "C", flags: rowMaxFlag(p.m) }),
                    cellN ? T.note(cellN.lines, { tex: cellN.tex }) : null,
                    T.note(["row sums: " + ex.out2.map(r => r.reduce((a, c) => a + c, 0).toFixed(2)).join("  ")])].filter(Boolean);
          } },
      ] };
  }

  function build(st) {
    switch (st.key) {
      case "patch_embed": return patchStages(st);
      case "extra_linear": return extraLinearStages(st);
      case "cls_token": return clsStages(st);
      case "pe": return peStages(st);
      case "enc_res1": case "enc_res2": return residualStages(st);
      case "cls_select": return clsSelectStages(st);
      case "head": return headStages(st);
      case "softmax": return softmaxStages(st);
      default: return null;
    }
  }

  function parallel(st) {
    const Bn = B.src.length, Tn = B.seq_total;
    switch (st.key) {
      case "patch_embed": return `One Conv2d call projects all ${Bn} × ${N} patches at once: as a matmul, (${Bn}·${N}, ${PPC}) × (${PPC}, ${d}). The patch-by-patch stages are only for explanation.`;
      case "extra_linear": return `One matmul for every patch of every image: (${Bn}·${N}, ${d}) × (${d}, ${d}).`;
      case "cls_token": return "expand makes a view (no copy) and torch.cat builds the whole (B, N + 1, D) tensor in one call.";
      case "pe": return `x + pe[:, :T] adds the (1, ${Tn}, ${d}) table to all ${Bn} images in one broadcast addition.`;
      case "enc_res1": case "enc_res2": return `One element-wise addition over the whole (${Bn}, ${Tn}, ${d}) tensor.`;
      case "cls_select": return "Indexing x[:, 0, :] takes token 0 of every image in one operation (a view).";
      case "head": return `One matmul for the whole batch: (${Bn}, ${d}) × (${d}, ${cfg.num_classes}).`;
      case "softmax": return `Softmax runs over the class axis for all ${Bn} images at once.`;
      default: return null;
    }
  }

  const ENC = ["enc_norm1", "enc_self_attn", "enc_res1", "enc_norm2", "enc_ffn", "enc_res2"];
  const crumbs = {
    short: { patch_embed: "Patches · E", extra_linear: "Extra Linear", cls_token: "[CLS]", pe: "+ E_pos",
             enc_norm1: "LN", enc_self_attn: "MSA", enc_res1: "+", enc_norm2: "LN", enc_ffn: "MLP", enc_res2: "+",
             enc_final: "Final LN", cls_select: "take [CLS]", head: "Head", softmax: "Softmax" },
    rows: [
      { label: "embed", items: ["patch_embed", "extra_linear", "cls_token", "pe", { group: "Enc", which: "enc", keys: ENC, N: cfg.n_encoder_layers }] },
      { label: "classify", items: ["enc_final", "cls_select", "head", "softmax"] },
    ],
  };

  return { build, parallel, crumbs };
};
