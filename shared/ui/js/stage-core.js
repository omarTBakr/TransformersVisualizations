/*
 * Operation stage (shared by every model): explains the operation running at the current step.
 *
 *   breadcrumb   the model's steps, with the running one lit (rows / groups come from the model)
 *   header       the operation name, centred
 *   tracking     the tensors in play (X, or Q / K / V for attention)
 *   visual       operands → operator → result, as isometrics with the real values
 *   equation     with the term for the current stage highlighted
 *   stages       ‹ 3 / 6 › title + one-sentence "how"
 *
 * Each stage maps to a simulator sub-step, so the code panel outlines the matching line.
 *
 * This file holds the engine and the builders every Transformer shares (attention, LayerNorm,
 * feed-forward, softmax, matmul / linear views). A model adds its own steps (embeddings, masks,
 * patches, heads…) and its breadcrumb in js/stage-model.js, which defines
 *
 *   window.StageModel = api => ({ build(st), crumbs: {short, rows}, parallel(st) })
 *
 * Depends on window.App (template.html), window.Tensors (tensors.js) and window.StageModel.
 */
(function () {
  "use strict";
  const App = window.App;
  const root = document.getElementById("stageRoot");
  if (!App || !root || !window.Tensors || !window.StageModel) return;

  const { STEPS, B, cfg } = App;
  const { esc, fmt, word } = App.helpers;
  const SPEC = App.DATA.spec || {};
  const T = Tensors.create({ B, word, esc, sample: () => App.state().sample });
  const d = cfg.d_model, h = cfg.n_heads, dk = d / h;
  const sample = () => App.state().sample;

  // ---------------------------------------------------------------- helpers
  const subIdx = (st, pred) => (st.subs || []).findIndex(pred);
  const subBy = (st, pred) => (st.subs || []).find(pred);
  const byLabel = re => s => re.test(s.label);
  const byVar = v => s => s.var === v;
  const hasVals = st => (st.subs || []).some(s => s.vals) || !!st.out_vals;
  const one = slice => ({ ...slice, v: [slice.v[sample()]] });      // just the selected sample
  const words = side => T.idsOf(side)[sample()].map(id => word(id, side));
  const nf = x => T.fmtNum(x);

  function cubeOf(slice, opts = {}) {                                  // slice: {v:[B][T][K], D}
    if (!slice) return null;
    const s = opts.single ? one(slice) : slice;
    const side = T.sideOfT(slice.v[0].length);
    return T.cube({
      v: s.v, front: opts.single ? 0 : sample(), depthLabels: i => `sample ${(opts.single ? sample() : i) + 1}`,
      depthName: opts.single ? "" : `batch · B = ${slice.v.length}`,
      xLabels: i => words(side)[i], xPad: i => T.idsOf(side)[sample()][i] === B.pad,
      xName: T.seqName(slice.v[0].length, slice.T), yName: `features · ${slice.D}`, ...opts,
    });
  }
  // Sub-step slice → drawing (3-D batch slices become cubes, 4-D become per-head cubes, 2-D grids).
  function subPart(st, i, opts = {}) {
    const s = st.subs && st.subs[i];
    if (!s) return null;
    if (!s.vals) return shapeOnly(s.shape, opts.title || s.label);
    if (s.vals.names.length === 3 && s.vals.names[0] === "batch" && opts.single) {
      const sv = s.vals;
      return cubeOf({ v: sv.v, D: sv.full[2] }, opts);
    }
    return T.fromSlice(s.vals, opts);
  }
  // Attention map for one head of the selected sample: rows = queries, cols = keys.
  function headGrid(st, i, head = 0, title = "") {
    const s = st.subs && st.subs[i];
    if (!s || !s.vals) return s ? shapeOnly(s.shape, title) : null;
    const v = s.vals.v[sample()][head];
    const qT = v.length, kT = v[0].length;
    const qs = T.sideOfT(qT), ks = T.sideOfT(kT);
    return T.valueGrid(v, r => words(qs)[r], c => words(ks)[c], s.vals.kind, { title: title || `head ${head + 1}, sample ${sample() + 1}` });
  }
  function shapeOnly(shape, title) {
    return T.note([title, fmt(shape), "values shown for layer 1"], { w: 190, cls: "dim" });
  }
  const eqTerm = (id, html) => `<span class="term" data-term="${id}">${html}</span>`;


  // ---------------------------------------------------------------- shared stage builders
  function attentionStages(st) {
    const cross = st.key === "dec_cross_attn", masked = st.key === "dec_masked_attn";
    const qp = subIdx(st, byVar("q_proj")), kp = subIdx(st, byVar("k_proj")), vp = subIdx(st, byVar("v_proj"));
    const dot = subIdx(st, byLabel(/^Dot products/)), addm = subIdx(st, byLabel(/^Add the mask/));
    const sm = subIdx(st, byLabel(/^Softmax/)), ws = subIdx(st, byLabel(/^Weighted sum/));
    const cat = subIdx(st, byLabel(/^Concatenate/)), op = subIdx(st, byVar("out_proj"));
    const maskOps = (st.subs || []).map((s, i) => [s, i]).filter(([s]) => /^(Boolean mask|Reshape padding|Merge causal)/.test(s.label)).map(([, i]) => i);
    const kvFrom = cross ? "encoder memory" : "X";
    const stages = [
      { title: "Project into Q, K and V", term: "proj", sub: qp,
        how: cross ? "Queries come from the decoder; keys and values are projections of the encoder memory."
                   : "The same input is multiplied by three learned matrices.",
        parts: () => [subPart(st, qp, { single: true, title: "Q", cw: 18 }), subPart(st, kp, { single: true, title: `K (from ${kvFrom})`, cw: 18 }),
                      subPart(st, vp, { single: true, title: `V (from ${kvFrom})`, cw: 18 })] },
      { title: `Split into ${h} heads`, term: "split", sub: qp + 1,
        how: `${d} features are reshaped into ${h} heads × ${dk} dims (view + transpose), so every head attends in parallel.`,
        parts: () => [subPart(st, qp + 1, { title: `Q split: ${h} heads (sample ${sample() + 1})` })] },
      { title: "Scores: Q·Kᵀ / √d_k", term: "qk", sub: dot,
        how: `Each query is compared with every key; dividing by √${dk} = ${Math.sqrt(dk).toFixed(0)} keeps the softmax out of its flat region.`,
        parts: () => [headGrid(st, dot, 0, "scores, head 1")] },
    ];
    if (addm >= 0) stages.push({ title: "Apply the mask", term: "m", sub: addm,
      how: masked ? "Future positions (causal mask) and padding get −∞, so softmax gives them weight 0."
                  : "Padding keys get −∞, so softmax gives them weight 0.",
      parts: () => [maskOps.length ? subPart(st, maskOps[maskOps.length - 1], { title: "mask (0 / −∞)" }) : null, T.op("+"), headGrid(st, addm, 0, "masked scores, head 1")].filter(Boolean) });
    stages.push({ title: "Softmax over the keys", term: "sm", sub: sm,
      how: "Each row becomes a probability distribution: non-negative and summing to 1.",
      parts: () => {
        const s = st.subs[sm];
        const rows = s && s.vals ? s.vals.v[sample()][0].map(r => r.reduce((a, b) => a + b, 0)) : [];
        return [headGrid(st, sm, 0, "attention weights, head 1"), rows.length ? T.note(["row sums:", rows.map(x => x.toFixed(2)).join(" ")], { w: 230 }) : null].filter(Boolean);
      } });
    stages.push({ title: "Weighted sum of V", term: "wv", sub: ws,
      how: "Each position's output is the attention-weighted average of the value vectors.",
      parts: () => [headGrid(st, sm, 0, "weights"), T.op("·"), subPart(st, vp + 1, { title: "V (heads)" }), T.op("="), subPart(st, ws, { title: "context (heads)" })] });
    stages.push({ title: "Concatenate the heads", term: "cat", sub: cat,
      how: `The ${h} heads (${dk} dims each) are joined back into ${d} features.`,
      parts: () => [subPart(st, cat, { single: true, title: "concat", heads: h, yName: `features · ${d} · ${h} heads` })] });
    stages.push({ title: "Output projection W_O", term: "wo", sub: op,
      how: "A final learned matrix mixes the heads together.",
      parts: () => [subPart(st, cat, { single: true, title: "concat" }), T.op("× W_O"), cubeOf(st.out_vals, { single: true, title: "output" })] });
    return {
      title: cross ? "Cross-Attention" : masked ? "Masked Multi-Head Self-Attention" : "Multi-Head Self-Attention",
      track: () => [["Q", st.subs[qp + 1]], ["K", st.subs[kp + 1], cross ? "from memory" : ""], ["V", st.subs[vp + 1], cross ? "from memory" : ""]]
        .filter(([, s]) => s).map(([n, s, note]) => ({ name: n, shape: s.shape, note, stage: 0 })),
      eq: `${eqTerm("proj", "Q = XW<sup>Q</sup>, K = " + (cross ? "MW" : "XW") + "<sup>K</sup>, V = " + (cross ? "MW" : "XW") + "<sup>V</sup>")}
           ${eqTerm("split", "→ h heads")}<br>
           head = ${eqTerm("sm", "softmax")}( ${eqTerm("qk", "QK<sup>T</sup> / √d<sub>k</sub>")} ${eqTerm("m", "+ M")} ) ${eqTerm("wv", "V")}<br>
           MultiHead = ${eqTerm("cat", "Concat(head<sub>1</sub>, …, head<sub>h</sub>)")} ${eqTerm("wo", "W<sup>O</sup>")}`,
      stages,
    };
  }

  // Post-norm models (the original Transformer) record a "Residual add" sub-step inside the
  // LayerNorm step; pre-norm models (ViT) normalize the plain input and add the residual later.
  function normStages(st) {
    const final = st.key.endsWith("final");
    const res = subIdx(st, byLabel(/^Residual add/)), xn = res >= 0 ? "x'" : "x", mu = subIdx(st, byLabel(/^Mean/)), va = subIdx(st, byLabel(/^Variance/));
    const nz = subIdx(st, byLabel(/^Normalize/)), sc = subIdx(st, byLabel(/^Scale and shift/));
    const xs = st.subs[res] && st.subs[res].vals ? st.subs[res].vals : null;
    const side = st.out_vals ? T.sideOfT(st.out_vals.v[0].length) : "src";
    const t = Math.min(1, (st.out_vals ? st.out_vals.v[0].length : 2) - 1);           // first real word
    const xSlice = xs ? { v: xs.v, D: xs.full[2] } : st.in_vals;
    const val = i => { const s = st.subs[i]; return s && s.vals ? s.vals.v[sample()][t] : null; };
    const stages = [
      { title: "Where LayerNorm acts", term: "all", sub: null,
        how: `For every (batch, position) pair, the ${d} features (one fibre) are normalized on their own.`,
        parts: () => [T.genericCube({ T: 6, K: 10, Dn: 2, xName: "sequence · T", yName: `features · d_model = ${d}`, depthName: "batch · B", hiCols: new Set([2]), title: "one highlighted fibre = one (b, t)" }),
                      T.note(["μ and σ² are taken", "along the feature axis", "of each fibre"], { w: 170 })] },
    ];
    if (res >= 0) stages.push({ title: "Residual add", term: "res", sub: res,
      how: "The sub-layer's output is added back to its input (the skip connection).",
      parts: () => [cubeOf(xSlice, { single: true, title: "x' = x + Sublayer(x)" })] });
    stages.push({ title: `Mean μ of “${words(side)[t]}”`, term: "mu", sub: mu,
      how: `Average the ${d} features of this position.`,
      parts: () => [cubeOf(xSlice, { single: true, hiCols: new Set([t]), title: xn }), T.op("mean", "over features"),
                    T.note([`μ = ${val(mu) != null ? nf(val(mu)) : "…"}`], { w: 120 })] });
    stages.push({ title: "Variance σ²", term: "var", sub: va,
      how: "Average squared distance from μ, over the same fibre.",
      parts: () => [cubeOf(xSlice, { single: true, hiCols: new Set([t]), title: xn }), T.op("var", "over features"),
                    T.note([`σ² = ${val(va) != null ? nf(val(va)) : "…"}`], { w: 120 })] });
    stages.push({ title: "Normalize", term: "norm", sub: nz,
      how: "Subtract μ and divide by √(σ² + ε): every fibre now has mean 0 and variance 1.",
      parts: () => [cubeOf(xSlice, { single: true, hiCols: new Set([t]), title: xn }), T.op("(x−μ)/σ"),
                    subPart(st, nz, { single: true, hiCols: new Set([t]), title: "x̂" })] });
    stages.push({ title: "Scale γ and shift β", term: "gb", sub: sc,
      how: "Learned per-feature γ and β let the model undo the normalization where useful.",
      parts: () => [subPart(st, nz, { single: true, title: "x̂" }), T.op("γ·x̂+β"), cubeOf(st.out_vals, { single: true, title: "y" })] });
    return {
      title: final ? "Final LayerNorm" : res >= 0 ? "Add & Norm" : "LayerNorm",
      eq: `${res < 0 ? "" : `x' = ${eqTerm("res", "x + Sublayer(x)")}<br>`}
           y = ${eqTerm("gb", "γ")} ⊙ ${eqTerm("norm", `(${xn} − ${eqTerm("mu", "μ")}) / √(${eqTerm("var", "σ²")} + ε)`)} + ${eqTerm("gb", "β")} ${eqTerm("all", "")}
           <div class="eq-sub">
             ${eqTerm("mu", `μ = <span class="frac">1/${d}</span> Σ<sub>j=1…${d}</sub> ${xn}<sub>j</sub>`)}
             <span class="eq-gap"></span>
             ${eqTerm("var", `σ² = <span class="frac">1/${d}</span> Σ<sub>j=1…${d}</sub> (${xn}<sub>j</sub> − μ)²`)}
           </div>`,
      stages,
    };
  }

  // The feed-forward activation, as configured: ReLU (original Transformer) or GELU (ViT).
  const erf = x => {                                   // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const sg = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
    return sg * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
  };
  const ACTS = {
    relu: { name: "ReLU", op: "max(0, ·)", eq: x => `max(0, ${x})`, f: x => Math.max(0, x),
            how: "Negative values are set to 0.", tex: x => `\\max(0,\ ${ts3(x)})` },
    gelu: { name: "GELU", op: "x·Φ(x)", eq: x => `GELU(${x})`, f: x => 0.5 * x * (1 + erf(x / Math.SQRT2)),
            how: "GELU(x) = x·Φ(x), where Φ is the standard normal CDF: large positive values pass almost unchanged, negative values are squashed towards 0 instead of being cut to 0.",
            tex: x => `${ts3(x)}\\cdot\\Phi(${t3(x)}) = ${ts3(x)}\\cdot\\tfrac{1}{2}\\left(1 + \\operatorname{erf}\\!\\left(\\tfrac{${t3(x)}}{\\sqrt{2}}\\right)\\right) = ${ts3(x)}\\times${t3(0.5 * (1 + erf(x / Math.SQRT2)))}` },
    silu: { name: "SiLU", op: "x·σ(x)", eq: x => `SiLU(${x})`, f: x => x / (1 + Math.exp(-x)),
            how: "SiLU(x) = x·σ(x): negative values are squashed towards 0.", tex: x => `${ts3(x)}\\cdot\\sigma(${t3(x)})` },
  };
  const ACT = ACTS[cfg.activation] || ACTS.relu;

  function ffnStages(st) {
    const l1 = subIdx(st, byVar("net[0]")), re = subIdx(st, byLabel(/activation$/)), l2 = subIdx(st, byVar("net[3]"));
    const zeros = () => {
      const s = st.subs[re];
      if (!s || !s.vals) return "";
      const f = s.vals.v.flat(2);
      return `${Math.round(100 * f.filter(x => x === 0).length / f.length)}% of the sampled values are now 0`;
    };
    return {
      title: SPEC.ffn_title || "Position-wise Feed-Forward",
      eq: `${SPEC.ffn_name || "FFN"}(x) = ${eqTerm("relu", ACT.eq(eqTerm("w1", "xW<sub>1</sub> + b<sub>1</sub>")))} ${eqTerm("w2", "W<sub>2</sub> + b<sub>2</sub>")} ${eqTerm("x", "")}`,
      stages: [
        { title: "Input", term: "x", sub: null, how: "The same small network is applied to every position independently.",
          parts: () => [cubeOf(st.in_vals, { single: true, title: "x" })] },
        { title: `Expand: ${d} → ${cfg.d_ff}`, term: "w1", sub: l1, how: "A linear layer widens every position's features.",
          parts: () => [cubeOf(st.in_vals, { single: true, title: "x", cw: 18 }), T.op("× W₁ + b₁", `${d} → ${cfg.d_ff}`), subPart(st, l1, { single: true, title: "hidden", cw: 18 })] },
        { title: ACT.name, term: "relu", sub: re, how: cfg.activation === "relu" ? `Negative values are set to 0. ${zeros()}.` : ACT.how,
          parts: () => [subPart(st, l1, { single: true, title: "before", cw: 18 }), T.op(ACT.op), subPart(st, re, { single: true, title: "after", cw: 18 })] },
        { title: `Project back: ${cfg.d_ff} → ${d}`, term: "w2", sub: l2, how: "A second linear layer returns to the model width.",
          parts: () => [subPart(st, re, { single: true, title: "hidden", cw: 18 }), T.op("× W₂ + b₂", `${cfg.d_ff} → ${d}`), cubeOf(st.out_vals, { single: true, title: "output", cw: 18 })] },
      ],
    };
  }

  function softmaxStages(st) {
    const terms = ["m", "e", "s", "p"];
    const stages = (st.subs || []).map((s, i) => ({
      title: s.label, term: terms[i] || "p", sub: i,
      how: s.formula ? `${s.formula}.` : "",
      parts: () => [subPart(st, i, { single: true, title: s.var || "", cw: 18 }),
                    i === 3 && st.topk ? T.note([`${SPEC.top_label || "most likely next words"}:`, ...st.topk[sample()].ids.slice(0, 3).map((id, j) =>
                      `${B.tgt_vocab[id] ?? "token " + id}: ${(st.topk[sample()].probs[j] * 100).toFixed(4)}%`)], { w: 200 }) : null].filter(Boolean),
    }));
    return { title: "Softmax (outside the model)",
      eq: `p<sub>i</sub> = ${eqTerm("e", "exp(z<sub>i</sub> − " + eqTerm("m", "m") + ")")} / ${eqTerm("s", "Σ<sub>j</sub> exp(z<sub>j</sub> − m)")} ${eqTerm("p", "")}`,
      stages };
  }

  // ================================================================ detail presets (tiny / small)
  // Every value is recorded, so each operation is shown as grids of real numbers for the
  // selected sample (and head), with arrows for the direction of the computation and the
  // arithmetic written out. Matmuls: click any result cell to see its row × column.
  const DETAIL = !!(App.preset && App.preset.detail);
  let headSel = 0;          // which attention head the detail views show
  const picks = {};         // stage index → picked cell {i, j} or row {r}
  const pickOf = (def) => picks[stage] || def;

  const tgtWord = id => B.tgt_vocab[id] ?? String(id);
  function axisLab(name, n) {
    if (["seq", "queries", "keys"].includes(name)) { const w = words(T.sideOfT(n)); return i => w[i] ?? String(i); }
    if (name === "vocab" || name === "classes") return i => tgtWord(i);
    if (name === "patches") return i => `p${i + 1}`;                 // ViT patch tokens (no [CLS])
    if (name === "hidden (d_ff)") return i => `h${i}`;
    if (name === "head features") return i => `d${i}`;
    if (name === "features") return i => `f${i}`;
    return i => String(i);
  }
  // A sub-step's values as a 2-D matrix for the selected sample (and head), with axis names.
  function mat(sub, head = headSel) {
    if (!sub || !sub.vals) return null;
    const { v, names, full } = sub.vals, b = sample();
    let m, nm, fl;
    if (names[0] === "batch" && names[1] === "heads") { m = v[b][Math.min(head, v[b].length - 1)]; nm = names.slice(2); fl = full.slice(2); }
    else if (names[0] === "batch" && names.length === 3) { m = v[b]; nm = names.slice(1); fl = full.slice(1); }
    else if (names[0] === "batch" && names.length === 2) {
      if (sub.shape.length === 3 && sub.shape[2] === 1) { m = v[b].map(x => [x]); nm = [names[1], "value"]; fl = [full[1], 1]; }
      else { m = [v[b]]; nm = ["sample", names[1]]; fl = [1, full[1]]; }
    } else if (names.length === 2) { m = v; nm = names; fl = full; }
    else return null;
    return { m, rowLab: nm[0] === "sample" ? () => `sample ${b + 1}` : axisLab(nm[0], fl[0]), colLab: axisLab(nm[1], fl[1]), names: nm, kind: sub.vals.kind };
  }
  const matOf = (m, rowName, colName) => ({ m, rowLab: axisLab(rowName, m.length), colLab: axisLab(colName, m[0].length), names: [rowName, colName], kind: "float" });
  const transpose = m => m[0].map((_, j) => m.map(r => r[j]));
  function G(M, opts = {}) { return M ? T.valueGrid(M.m, M.rowLab, M.colLab, M.kind, opts) : null; }
  const argmaxAbs = m => { let bi = 0, bj = 0; m.forEach((r, i) => r.forEach((x, j) => { if (Math.abs(x) < 1e29 && Math.abs(x) > Math.abs(m[bi][bj])) { bi = i; bj = j; } })); return { i: bi, j: bj }; };
  const rowMaxFlag = m => (i, j, x) => (Math.abs(x) < 1e29 && x === Math.max(...m[i].filter(y => Math.abs(y) < 1e29)) ? "max" : "");
  const sign = x => (x < 0 ? `(${nf(x)})` : nf(x));
  // LaTeX helpers for the worked arithmetic (rendered with KaTeX below the equation).
  // Text inside LaTeX: escape specials; "·" and "^" have no text-mode glyph in KaTeX, so they drop to math.
  const txt = x => `\\text{${String(x).replace(/[\\{}$&#%_~]/g, c => "\\" + c)
    .replace(/·/g, "}\\cdot\\text{").replace(/\^/g, "}{}^\\wedge\\text{")}}`;
  const texName = n => String(n).replace(/ᵀ/g, "^{\\top}").replace(/₁/g, "_1").replace(/₂/g, "_2")
    .replace(/^x_p$/, "x^{(p)}").replace(/^x_p$/, "x^{(p)}").replace(/^W_([QKVO])/, "W_{$1}").replace(/^(context|score|weights|concat|logit|out|memory)$/, m => txt(m));
  const t3 = x => (Math.abs(x) >= 1e29 ? "-\\infty" : x.toFixed(3));
  const ts3 = x => (x < 0 ? `(${t3(x)})` : t3(x));
  // Worked arithmetic uses 3 decimals so a learner redoing it by hand gets the shown result.
  const n3 = x => (Math.abs(x) >= 1e29 ? "−∞" : x.toFixed(3));
  const s3 = x => (x < 0 ? `(${n3(x)})` : n3(x));

  /*
   * Matmul view: A (rows × k) · B (k × cols) = C, with the picked cell's row of A (→), column
   * of B (↓) and cell of C highlighted, and the dot product written out with the real numbers.
   */
  function matmul({ A, Bm, Cm, bias = null, scale = 1, tA = "A", tB = "B", tC = "C", how = "", rowFlags = null }) {
    if (!A || !Bm || !Cm) return { parts: () => [T.note(["values not recorded"], { w: 180, cls: "dim" })], note: "" };
    const sel = pickOf(argmaxAbs(Cm.m));
    const { i, j } = sel;
    const K = A.m[0].length;
    const terms = A.m[i].map((a, k) => [a, Bm.m[k][j]]);
    let dot = terms.reduce((s, [a, b]) => s + a * b, 0);
    let val = dot * scale + (bias ? bias[j] : 0);
    const shown = terms.slice(0, 6).map(([a, b]) => `${n3(a)}×${s3(b)}`).join(" + ") + (K > 6 ? ` + … (${K} terms)` : "");
    const lines = [
      `${tC}[${Cm.rowLab(i)}, ${Cm.colLab(j)}] = Σ ${tA}[${A.rowLab(i)}, k] · ${tB}[k, ${Cm.colLab(j)}]${scale !== 1 ? ` × ${n3(scale)}` : ""}${bias ? ` + b[${Cm.colLab(j)}]` : ""}`,
      `= (${shown})${scale !== 1 ? ` × ${n3(scale)}` : ""}${bias ? ` + ${s3(bias[j])}` : ""}`,
      `= ${n3(val)}   ${Math.abs(val - Cm.m[i][j]) < 0.02 + 0.01 * Math.abs(val) ? "✓ matches the model" : `(model: ${n3(Cm.m[i][j])})`}`,
    ];
    const idx = (r, c) => `{${txt(r)},\\,${txt(c)}}`;
    const shownT = terms.slice(0, 6).map(([a, b]) => `${t3(a)}\\times${ts3(b)}`).join(" + ") + (K > 6 ? ` + \\cdots\\ (${K}\\ ${txt("terms")})` : "");
    const ok = Math.abs(val - Cm.m[i][j]) < 0.02 + 0.01 * Math.abs(val);
    const tex = [
      `{${texName(tC)}}_${idx(Cm.rowLab(i), Cm.colLab(j))} = \\sum_{k} {${texName(tA)}}_{${txt(A.rowLab(i))},k}\\; {${texName(tB)}}_{k,${txt(Cm.colLab(j))}}${scale !== 1 ? `\\cdot ${t3(scale)}` : ""}${bias ? ` + b_{${txt(Cm.colLab(j))}}` : ""}`,
      `= \\left(${shownT}\\right)${scale !== 1 ? `\\cdot ${t3(scale)}` : ""}${bias ? ` + ${ts3(bias[j])}` : ""} = ${t3(val)}\\quad ${ok ? `\\checkmark\\ ${txt("matches the model")}` : txt(`(model: ${n3(Cm.m[i][j])})`)}`,
    ];
    return {
      parts: () => [
        G(A, { title: tA, hiRow: i, rowArrow: true, flags: rowFlags }), T.op("·"),
        G(Bm, { title: tB, hiCol: j, colArrow: true }), T.op("="),
        G(Cm, { title: `${tC} (click a cell)`, hiCell: { i, j }, gridId: "C", flags: rowFlags }),
        T.note(lines, { cls: "wide", tex }),
      ],
      how,
    };
  }
  // Linear layer sub-step: y = x · Wᵀ + b, using the recorded operands.
  function linearStage(st, i, title, names, how) {
    const s = st.subs[i];
    if (!s || !s.ops) return { title, term: names.term, sub: i, how, parts: () => [subPart(st, i, { single: true, title })] };
    const inName = names.in || "features", outName = names.out || "features";
    const outPre = (names.c || "y")[0].toLowerCase();
    // Built on every draw, so a clicked cell (and the selected sample) is picked up.
    const parts = () => {
      const b = sample();
      // Output columns are labelled after the result (q0, k0, …) so they don't look like the input features.
      const Bm = matOf(transpose(s.ops.W), inName, outName), Cm = matOf(s.vals.v[b], names.rows || "seq", outName);
      Bm.colLab = Cm.colLab = j => `${outPre}${j}`;
      return matmul({ A: matOf(s.ops.x[b], names.rows || "seq", inName), Bm, Cm, bias: s.ops.b,
                      tA: names.a || "x", tB: names.b || "Wᵀ", tC: names.c || "y" }).parts();
    };
    return { title, term: names.term, sub: i, how: how + " PyTorch stores W as (out × in), so the code computes x · Wᵀ. Click any result cell to see its dot product.", parts, pick: true };
  }

  // p[i][j] = exp(s_ij − m_i) / Σ_k exp(s_ik − m_i), written out with the real numbers (−∞ terms → 0).
  function softmaxCell(scoreRow, i, j, rowName, colName, pName = "w", sName = "s", model = null) {
    const fin = scoreRow.filter(x => Math.abs(x) < 1e29);
    if (!fin.length) return { lines: [`Every key in row ${rowName} is masked, so its weights are set to 0 (nan_to_num).`], tex: null };
    const m = Math.max(...fin);
    const ex = scoreRow.map(x => (Math.abs(x) >= 1e29 ? 0 : Math.exp(x - m)));
    const den = ex.reduce((a, b) => a + b, 0), val = ex[j] / den;
    const terms = scoreRow.map(x => (Math.abs(x) >= 1e29 ? "e^{-\\infty}" : `e^{${t3(x - m)}}`));
    const shown = terms.length > 8 ? [...terms.slice(0, 7), "\\cdots"] : terms;
    const ix = `{${txt(rowName)},\\,${txt(colName)}}`;
    return {
      lines: [`${pName}[${rowName}, ${colName}] = exp(${sName} − m) / Σ exp(…) = ${n3(val)}`],
      tex: [
        `${pName}_${ix} = \\frac{e^{\\,${sName}_${ix} - m}}{\\sum_k e^{\\,${sName}_{${txt(rowName)},k} - m}},\\qquad m = \\max_k ${sName}_{${txt(rowName)},k} = ${t3(m)}`,
        `= \\frac{${Math.abs(scoreRow[j]) >= 1e29 ? "e^{-\\infty}" : `e^{${t3(scoreRow[j] - m)}}`}}{${shown.join(" + ")}} = \\frac{${t3(ex[j])}}{${t3(den)}} = ${t3(val)}${model != null ? `\\quad ${txt(`(model: ${n3(model)})`)}` : ""}`,
      ],
    };
  }

  function attentionD(st) {
    const base = attentionStages(st);
    const cross = st.key === "dec_cross_attn";
    const qp = subIdx(st, byVar("q_proj")), kp = subIdx(st, byVar("k_proj")), vp = subIdx(st, byVar("v_proj"));
    const dot = subIdx(st, byLabel(/^Dot products/)), addm = subIdx(st, byLabel(/^Add the mask/));
    const sm = subIdx(st, byLabel(/^Softmax/)), nz = subIdx(st, byLabel(/^Zero any/)), ws = subIdx(st, byLabel(/^Weighted sum/));
    const cat = subIdx(st, byLabel(/^Concatenate/)), op = subIdx(st, byVar("out_proj"));
    const maskIdx = [...(st.subs || []).keys()].filter(i => /^(Merge causal|Reshape padding)/.test(st.subs[i].label));
    const kvName = cross ? "memory" : "x";
    const S = st.subs;
    const head = () => headSel;
    const qSide = st.key.startsWith("enc") ? "src" : "tgt";
    const padQ = i => T.idsOf(qSide)[sample()][i] === B.pad;
    const anyPadQ = () => T.idsOf(qSide)[sample()].includes(B.pad);
    const withPad = base => (i, j, x) => { const f = base ? base(i, j, x) : ""; return padQ(i) ? (f ? `${f} flag-padq` : "padq") : f; };
    const padNote = () => (anyPadQ() ? " Dimmed rows are <pad> queries: they are computed too, but nothing uses them later (no position attends to a pad key, and the loss skips pad targets)." : "");
    const stages = [
      linearStage(st, qp, "Q = x · W_Qᵀ + b", { term: "proj", a: "x", b: "W_Qᵀ", c: "Q" }, "Each query is a weighted mix of the input features."),
      linearStage(st, kp, `K = ${kvName} · W_Kᵀ + b`, { term: "proj", a: kvName, b: "W_Kᵀ", c: "K" }, cross ? "Keys come from the encoder memory." : "Keys use their own learned matrix."),
      linearStage(st, vp, `V = ${kvName} · W_Vᵀ + b`, { term: "proj", a: kvName, b: "W_Vᵀ", c: "V" }, cross ? "Values come from the encoder memory." : "Values use a third learned matrix."),
      { title: `Split into ${h} head${h > 1 ? "s" : ""}`, term: "split", sub: qp + 1,
        how: h > 1 ? `The ${d} features are cut into ${h} groups of ${dk}: head 1 gets f0–f${dk - 1}, head 2 the next ${dk}, …` : `With 1 head, the head gets all ${d} features.`,
        parts: () => Array.from({ length: h }, (_, hh) => G(mat(S[qp + 1], hh), { title: `Q, head ${hh + 1}${hh === head() ? " (selected)" : ""}` })) },
      (() => { const scale = 1 / Math.sqrt(dk); return { title: "Scores = Q · Kᵀ / √d_k", term: "qk", sub: dot, pick: true,
        get how() { return `Dot product of each query row with each key row, divided by √${dk}. Click a score.${padNote()}`; },
        parts: () => matmul({ A: mat(S[qp + 1], head()), Bm: (() => { const k = mat(S[kp + 1], head()); return k ? matOf(transpose(k.m), "head features", "keys") : null; })(),
                              Cm: mat(S[dot], head()), scale, tA: "Q", tB: "Kᵀ", tC: "score", rowFlags: withPad(null) }).parts() }; })(),
    ];
    if (addm >= 0) stages.push({ title: "Add the mask", term: "m", sub: addm,
      how: "Blocked positions (padding, or the future in the decoder) become −∞.",
      parts: () => [G(mat(S[dot], head()), { title: "scores", flags: withPad(null) }), T.op("+"), maskIdx.length ? G(mat(S[maskIdx[maskIdx.length - 1]], head()), { title: "mask" }) : null,
                    T.op("="), G(mat(S[addm], head()), { title: "masked scores", flags: withPad(null) })].filter(Boolean) });
    stages.push({ title: "Softmax over each row", term: "sm", sub: sm, pick: true,
      get how() { return `Each row is exponentiated and divided by its sum; the largest weight in each row is marked ★. Click a weight to see its calculation.${padNote()}`; },
      parts: () => { const w = mat(S[nz >= 0 ? nz : sm], head()), sc = mat(S[addm >= 0 ? addm : dot], head());
        const { i, j } = pickOf({ i: 0, j: w.m[0].indexOf(Math.max(...w.m[0])) });
        const cellN = softmaxCell(sc.m[i], i, j, w.rowLab(i), w.colLab(j), "w", "s", w.m[i][j]);
        return [G(sc, { title: "scores", hiRow: i, rowArrow: true, flags: withPad(null) }), T.op("softmax", "per row"),
                G(w, { title: "weights (click a cell)", hiCell: { i, j }, gridId: "C", flags: withPad(rowMaxFlag(w.m)) }),
                T.note(cellN.lines, { tex: cellN.tex }),
                T.note(["row sums: " + w.m.map(r => r.reduce((a, b) => a + b, 0).toFixed(2)).join("  ")])]; } });
    stages.push({ title: "Context = weights · V", term: "wv", sub: ws, pick: true,
      get how() { return `Each output row is the weighted average of the value rows. Click a cell.${padNote()}`; },
      parts: () => matmul({ A: mat(S[nz >= 0 ? nz : sm], head()), Bm: mat(S[vp + 1], head()), Cm: mat(S[ws], head()), tA: "weights", tB: "V", tC: "context", rowFlags: withPad(null) }).parts() });
    stages.push({ title: "Concatenate the heads", term: "cat", sub: cat,
      how: `The ${h} head outputs are placed side by side to get back ${d} features.`,
      parts: () => [...Array.from({ length: h }, (_, hh) => G(mat(S[ws], hh), { title: `head ${hh + 1}` })), T.op("→"), G(mat(S[cat]), { title: "concat" })] });
    stages.push(linearStage(st, op, "Output = concat · W_Oᵀ + b", { term: "wo", a: "concat", b: "W_Oᵀ", c: "out" }, "A last matrix mixes the heads."));
    return { ...base, stages, heads: h };
  }

  function normD(st) {
    const res = subIdx(st, byLabel(/^Residual add/)), xn = res >= 0 ? "x'" : "x", mu = subIdx(st, byLabel(/^Mean/)), va = subIdx(st, byLabel(/^Variance/));
    const nzI = subIdx(st, byLabel(/^Normalize/)), sc = subIdx(st, byLabel(/^Scale and shift/));
    const S = st.subs;
    const Xs = res >= 0 ? mat(S[res]) : matOf(st.in_vals.v[sample()], "seq", "features");
    const rowPick = () => pickOf({ i: Math.min(1, Xs.m.length - 1) }).i;
    const muV = r => mat(S[mu]).m[r][0], vaV = r => mat(S[va]).m[r][0];
    const g = S[sc] && S[sc].ops, eps = g ? g.eps : 1e-5;
    const base = normStages(st);
    const stages = [base.stages[0]];
    if (res >= 0) stages.push({ title: "Residual add", term: "res", sub: res, how: "x' = x + Sublayer(x): the skip connection.", parts: () => [G(Xs, { title: "x'" })] });
    stages.push({ title: "Mean μ of a row", term: "mu", sub: mu, pick: true,
      how: "Average the features of one position (click a row to pick another).",
      parts: () => { const r = rowPick(), row = Xs.m[r];
        return [G(Xs, { title: `${xn} (click a row)`, hiRow: r, rowArrow: true, gridId: "C" }), T.op("mean →"),
                T.note([`μ = (${row.map(s3).join(" + ")}) / ${row.length}`, `= ${n3(row.reduce((a, b) => a + b, 0) / row.length)}   (model: ${n3(muV(r))})`],
                  { tex: [`\\mu = \\frac{${row.map(ts3).join(" + ")}}{${row.length}} = ${t3(row.reduce((a, b) => a + b, 0) / row.length)}\\quad ${txt(`(model: ${n3(muV(r))})`)}`] })]; } });
    stages.push({ title: "Variance σ²", term: "var", sub: va, pick: true,
      how: "Average squared distance from μ along the same row.",
      parts: () => { const r = rowPick(), row = Xs.m[r], m = muV(r);
        return [G(Xs, { title: `${xn} (click a row)`, hiRow: r, rowArrow: true, gridId: "C" }), T.op("var →"),
                T.note([`μ = ${n3(m)}`, `σ² = (${row.slice(0, 6).map(x => `(${n3(x)} − ${s3(m)})²`).join(" + ")}${row.length > 6 ? " + …" : ""}) / ${row.length}`, `= ${n3(row.reduce((a, b) => a + (b - m) ** 2, 0) / row.length)}   (model: ${n3(vaV(r))})`],
                  { tex: [`\\mu = ${t3(m)}`, `\\sigma^2 = \\frac{${row.slice(0, 6).map(x => `(${t3(x)} - ${ts3(m)})^2`).join(" + ")}${row.length > 6 ? " + \\cdots" : ""}}{${row.length}} = ${t3(row.reduce((a, b) => a + (b - m) ** 2, 0) / row.length)}\\quad ${txt(`(model: ${n3(vaV(r))})`)}`] })]; } });
    stages.push({ title: "Normalize", term: "norm", sub: nzI, pick: true,
      how: `x̂ = (${xn} − μ) / √(σ² + ε): this row now has mean 0 and variance 1.`,
      parts: () => { const r = rowPick(), m = muV(r), sd = Math.sqrt(vaV(r) + eps), xh = mat(S[nzI]);
        return [G(Xs, { title: `${xn} (click a row)`, hiRow: r, gridId: "C" }), T.op("(x−μ)/σ"), G(xh, { title: "x̂", hiRow: r }),
                T.note([`μ = ${n3(m)},  σ = √(σ² + ε) = ${n3(sd)}`, ...Xs.m[r].slice(0, 6).map((x, j) => `x̂${j} = (${n3(x)} − ${s3(m)}) / ${n3(sd)} = ${n3((x - m) / sd)}`)],
                  { tex: [`\\mu = ${t3(m)},\\qquad \\sigma = \\sqrt{\\sigma^2 + \\varepsilon} = ${t3(sd)}`,
                          ...Xs.m[r].slice(0, 6).map((x, j) => `\\hat{x}_{${j}} = \\frac{${t3(x)} - ${ts3(m)}}{${t3(sd)}} = ${t3((x - m) / sd)}`)] })]; } });
    stages.push({ title: "Scale γ and shift β", term: "gb", sub: sc, pick: true,
      how: "y = γ ⊙ x̂ + β with a learned γ and β per feature. Click a cell.",
      parts: () => { const xh = mat(S[nzI]), y = mat(S[sc]); const { i, j } = pickOf({ i: rowPick(), j: 0 });
        const untrained = g && g.gamma.every(x => Math.abs(x - 1) < 1e-6) && g.beta.every(x => Math.abs(x) < 1e-6);
        const lines = g ? [`γ = [${g.gamma.map(n3).join(", ")}]`, `β = [${g.beta.map(n3).join(", ")}]`,
          `y = ${n3(g.gamma[j])} × ${s3(xh.m[i][j])} + ${s3(g.beta[j])} = ${n3(g.gamma[j] * xh.m[i][j] + g.beta[j])}`,
          ...(untrained ? ["γ = 1 and β = 0 because the model is untrained, so here y = x̂ exactly."] : [])] : [];
        const tex = g ? [`\\gamma = [${g.gamma.map(t3).join(",\\ ")}],\\qquad \\beta = [${g.beta.map(t3).join(",\\ ")}]`,
          `y_{${j}} = \\gamma_{${j}}\\,\\hat{x}_{${j}} + \\beta_{${j}} = ${t3(g.gamma[j])}\\times${ts3(xh.m[i][j])} + ${ts3(g.beta[j])} = ${t3(g.gamma[j] * xh.m[i][j] + g.beta[j])}`,
          ...(untrained ? [txt("γ = 1 and β = 0 because the model is untrained, so here y = x̂ exactly.")] : [])] : null;
        return [G(xh, { title: "x̂", hiCell: { i, j } }), T.op("γ⊙x̂+β"), G(y, { title: "y (click a cell)", hiCell: { i, j }, gridId: "C" }), lines.length ? T.note(lines, { tex }) : null].filter(Boolean); } });
    return { ...base, stages };
  }

  // Recomputed from values shown to 3 significant figures, so allow the same tolerance as the matmul view.
  const close = (v, m) => Math.abs(v - m) < 0.02 + 0.01 * Math.abs(m);
  const checkTxt = (v, m) => (close(v, m) ? `✓ matches the model (${n3(m)})` : `(model: ${n3(m)})`);
  const checkTex = (v, m) => (close(v, m) ? `\\checkmark\\ ${txt(`matches the model (${n3(m)}; inputs shown rounded)`)}` : txt(`(model: ${n3(m)})`));

  // GELU / SiLU: every cell changes, so show one picked cell's arithmetic instead of counting zeros.
  function smoothActD(st, re) {
    const S = st.subs;
    return { title: `${ACT.name}: ${ACT.op}`, term: "relu", sub: re, pick: true,
      how: `${ACT.how} Click a cell to see its value worked out.`,
      parts: () => {
        const before = S[re] && S[re].ops ? matOf(S[re].ops.x[sample()], "seq", "hidden (d_ff)") : null, after = mat(S[re]);
        if (!before || !after) return [subPart(st, re, { single: true, title: "after" })];
        const { i, j } = pickOf(argmaxAbs(before.m));
        const x = before.m[i][j], y = ACT.f(x);
        return [G(before, { title: "before", hiCell: { i, j }, flags: (r, c, v) => (v < 0 ? "neg" : "") }), T.op(ACT.op),
                G(after, { title: "after (click a cell)", hiCell: { i, j }, gridId: "C" }),
                T.note([`${ACT.name}(${n3(x)}) = ${n3(y)}   ${checkTxt(y, after.m[i][j])}`],
                  { tex: [`\\operatorname{${ACT.name}}(${t3(x)}) = ${ACT.tex(x)} = ${t3(y)}\\quad ${checkTex(y, after.m[i][j])}`] })];
      } };
  }

  function ffnD(st) {
    const l1 = subIdx(st, byVar("net[0]")), re = subIdx(st, byLabel(/activation$/)), l2 = subIdx(st, byVar("net[3]"));
    const S = st.subs;
    return { ...ffnStages(st), stages: [
      linearStage(st, l1, `Expand: h = x · W₁ᵀ + b₁ (${d} → ${cfg.d_ff})`, { term: "w1", a: "x", b: "W₁ᵀ", c: "h", out: "hidden (d_ff)" }, "Widen each position's features."),
      cfg.activation !== "relu" ? smoothActD(st, re) : { title: "ReLU: negatives become 0", term: "relu", sub: re,
        how: "Cells that were negative (red) are set to 0 (mint); the largest value is marked ★.",
        parts: () => { const before = S[re] && S[re].ops ? matOf(S[re].ops.x[sample()], "seq", "hidden (d_ff)") : null, after = mat(S[re]);
          const zeroed = before ? before.m.flat().filter(x => x < 0).length : 0;
          const mx = after ? Math.max(...after.m.flat()) : 0;
          return [G(before, { title: "before", flags: (i, j, x) => (x < 0 ? "neg" : "") }), T.op("max(0, ·)"),
                  G(after, { title: "after", flags: (i, j, x) => (before && before.m[i][j] < 0 ? "zero" : x === mx ? "max" : "") }),
                  T.note([`${zeroed} of ${before ? before.m.flat().length : 0} values zeroed`, `max = ${nf(mx)}`], { w: 200 })]; } },
      linearStage(st, l2, `Project back: y = h · W₂ᵀ + b₂ (${cfg.d_ff} → ${d})`, { term: "w2", a: "h", b: "W₂ᵀ", c: "y", in: "hidden (d_ff)" }, "Return to the model width."),
    ] };
  }

  function softmaxD(st) {
    const S = st.subs, b = sample();
    const Z = matOf(st.in_vals.v[b], "seq", "vocab");
    const m = mat(S[0]), e = mat(S[1]), sum = mat(S[2]), p = mat(S[3]);
    return { ...softmaxStages(st), stages: [
      { title: "Row max m (for stability)", term: "m", sub: 0, how: "The largest logit in each row is marked ★ and subtracted before exp, so exp can't overflow.",
        parts: () => [G(Z, { title: "logits z", flags: rowMaxFlag(Z.m) }), T.op("max →"), G(m, { title: "m" })] },
      { title: "Exponentiate e = exp(z − m)", term: "e", sub: 1, how: "The row max becomes exp(0) = 1; everything else is smaller.",
        parts: () => [G(Z, { title: "z" }), T.op("exp(z−m)"), G(e, { title: "e", flags: rowMaxFlag(e.m) })] },
      { title: `Sum Σ over the ${SPEC.out_axis || "vocabulary"}`, term: "s", sub: 2, how: "Add up each row.",
        parts: () => [G(e, { title: "e", rowArrow: true, hiRow: 0 }), T.op("Σ →"), G(sum, { title: "Σ" })] },
      { title: "Divide p = e / Σ", term: "p", sub: 3, pick: true, how: `Each row now sums to 1; the most likely ${SPEC.out_item || "next word"} is marked ★. Click a probability to see its calculation.`,
        parts: () => { const { i, j } = pickOf({ i: 0, j: p.m[0].indexOf(Math.max(...p.m[0])) });
          const cellN = softmaxCell(Z.m[i], i, j, p.rowLab(i), p.colLab(j), "p", "z", p.m[i][j]);
          return [G(p, { title: "p (click a cell)", hiCell: { i, j }, gridId: "C", flags: rowMaxFlag(p.m) }),
                  T.note(cellN.lines, { tex: cellN.tex }), T.note(["row sums: " + p.m.map(r => r.reduce((a, c) => a + c, 0).toFixed(2)).join("  ")])]; } },
    ] };
  }

  // ---------------------------------------------------------------- model hook-up
  const API = {
    App, T, cfg, B, STEPS, SPEC, d, h, dk, esc, fmt, word, sample,
    subIdx, subBy, byLabel, byVar, hasVals, one, words, nf, cubeOf, subPart, headGrid, shapeOnly, eqTerm,
    mat, matOf, transpose, G, argmaxAbs, rowMaxFlag, sign, txt, texName, t3, ts3, n3, s3,
    matmul, linearStage, softmaxCell, pickOf: def => pickOf(def),
    attentionStages, attentionD, normStages, normD, ffnStages, ffnD, softmaxStages, softmaxD,
    checkTxt, checkTex,
    get DETAIL() { return DETAIL; },
  };
  const M = window.StageModel(API);

  // Shared steps are recognised by their key; the model handles everything else first.
  function genericBuild(st) {
    if (/attn$/.test(st.key)) return DETAIL ? attentionD(st) : attentionStages(st);
    if (/norm\d$|_final$/.test(st.key)) return DETAIL ? normD(st) : normStages(st);
    if (/ffn$/.test(st.key)) return DETAIL ? ffnD(st) : ffnStages(st);
    if (st.key === "softmax") return DETAIL ? softmaxD(st) : softmaxStages(st);
    return null;
  }
  function buildFor(st) {
    return (M.build && M.build(st)) || genericBuild(st)
      || { title: st.title, eq: "", stages: [{ title: st.title, term: "", sub: null, how: st.desc, parts: () => [] }] };
  }

  // "Everything at once": what the code computes in a single tensor operation for this step.
  function parallelNote(st) {
    const own = M.parallel && M.parallel(st);
    if (own != null) return own;
    const Bn = B.src.length, Tn = st.out_shape && st.out_shape.length > 1 ? st.out_shape[1] : 0;
    const shp = s => (s ? fmt(s.shape) : "");
    if (/attn$/.test(st.key)) {
      const qi = subIdx(st, byVar("q_proj")), ki = subIdx(st, byVar("k_proj"));
      const qs = st.subs[qi + 1], ks = st.subs[ki + 1], sc = subBy(st, byLabel(/^Dot products/));
      return `Every query, key, head and sample is handled together: Q, K and V are three (${Bn}·${Tn}, ${d}) × (${d}, ${d}) matrix multiplications, `
        + `and Q·Kᵀ is one batched matmul ${shp(qs)} × ${ks ? fmt([...ks.shape.slice(0, 2), ks.shape[3], ks.shape[2]]) : ""} → ${shp(sc)}, ${h > 1 ? `so all ${h} heads run in parallel.` : "for the single head."}`;
    }
    if (/norm\d$|_final$/.test(st.key)) return `μ, σ² and the normalization are computed for all ${Bn} × ${Tn} = ${Bn * Tn} fibres in one vectorized call (nn.LayerNorm); the single highlighted position is just an example.`;
    if (/ffn$/.test(st.key)) return `Each linear layer is one matrix multiplication over every position at once: (${Bn}·${Tn}, ${d}) × (${d}, ${cfg.d_ff}), then (${Bn}·${Tn}, ${cfg.d_ff}) × (${cfg.d_ff}, ${d}).`;
    if (st.key === "softmax") return `Softmax runs over the ${SPEC.out_axis || "vocabulary"} axis for all ${Bn} × ${Tn} positions at once.`;
    return "";
  }

  // ---------------------------------------------------------------- breadcrumb
  // M.crumbs = { short: {key: label}, rows: [{ label, items: [key | {group, which, keys, N}], tail }] }
  const stepIndex = (key, layer) => STEPS.findIndex(s => s.key === key && (s.layer == null || layer == null || s.layer === layer));
  const GROUPS = M.crumbs.rows.flatMap(r => r.items.filter(it => typeof it === "object"));
  const lastLayer = Object.fromEntries(GROUPS.map(g => [g.which, 1]));

  function crumbs(st) {
    const C = M.crumbs, sep = `<span class="csep">›</span>`;
    const L = which => (st.key.startsWith(which + "_") && st.layer ? st.layer : lastLayer[which]);
    const chip = (key, layer = null) => `<button type="button" class="crumb${st.key === key ? " on" : ""}" data-key="${key}" data-layer="${layer ?? ""}">${C.short[key]}</button>`;
    const item = it => (typeof it === "string" ? chip(it) : `<span class="cgroup${it.keys.includes(st.key) ? " on" : ""}">
        <span class="cname">${it.group}</span><button type="button" class="clayer" data-which="${it.which}" data-d="-1">‹</button><b>${L(it.which)}/${it.N}</b><button type="button" class="clayer" data-which="${it.which}" data-d="1">›</button>
        ${it.keys.map(k => chip(k, L(it.which))).join("")}</span>`);
    return C.rows.map(r => `<span class="crow"><span class="clabel">${r.label}</span>${r.items.map(item).join(sep)}${r.tail ? `<span class="csep">${r.tail}</span>` : ""}</span>`).join("\n");
  }

  // ---------------------------------------------------------------- render
  root.innerHTML = `
    <div class="st-top"><div class="st-crumbs" id="stCrumbs"></div><span class="seg" id="stSample" role="group" aria-label="Sample shown"></span></div>
    <div class="st-head"><div class="st-title" id="stTitle"></div><div class="st-layer" id="stLayer"></div><div class="st-track" id="stTrack"></div></div>
    <div class="st-par" id="stPar"></div>
    <div class="st-visual"><svg class="tensor st-svg" id="stSvg" role="img" aria-label="The current operation, as isometric tensors"></svg></div>
    <div class="st-eq" id="stEq"></div>
    <div class="st-work" id="stWork" hidden></div>
    <div class="st-ctrl">
      <button type="button" class="pbtn" data-sd="-1" title="Previous stage (Shift+←)">‹</button>
      <span class="st-count" id="stCount"></span>
      <button type="button" class="pbtn" data-sd="1" title="Next stage (Shift+→)">›</button>
      <span class="st-dots" id="stDots"></span>
      <span class="st-headsel" id="stHeadSel"></span>
    </div>
    <div class="st-how"><b id="stStageTitle"></b> <span id="stHow"></span></div>
    <p class="st-foot" id="stFoot"></p>`;
  const $ = id => document.getElementById(id);
  let model = null, stage = 0, forStep = -1;

  function draw() {
    const { cur } = App.state();
    const st = STEPS[cur];
    if (!model) return;
    const S = model.stages[stage];
    $("stCrumbs").innerHTML = crumbs(st);
    $("stSample").innerHTML = B.src.map((_, b) => `<button type="button" data-sample="${b}" class="${b === sample() ? "on" : ""}">sample ${b + 1}</button>`).join("");
    $("stTitle").innerHTML = esc(model.title);
    // Which encoder / decoder layer is being computed right now.
    const which = st.key.startsWith("enc_") ? "Encoder" : st.key.startsWith("dec_") ? "Decoder" : "";
    $("stLayer").innerHTML = st.layer && which
      ? `${which} layer <b>${st.layer}</b> of ${st.nlayers} <span class="ldots">${Array.from({ length: st.nlayers }, (_, i) =>
          `<i class="${i + 1 < st.layer ? "done" : i + 1 === st.layer ? "now" : ""}"></i>`).join("")}</span>
          <span class="lnote">layers run one after another: each needs the previous layer's output</span>`
      : which && st.key.endsWith("final") ? `${which}: after all ${st.key.startsWith("enc") ? cfg.n_encoder_layers : cfg.n_decoder_layers} layers` : "";
    const par = parallelNote(st);
    $("stPar").innerHTML = par ? `<b>⚡ All at once</b> ${esc(par)}` : "";
    $("stPar").hidden = !par;
    const track = model.track ? model.track() : [{ name: "in", shape: (st.inputs.find(i => i.shape) || {}).shape }, { name: "out", shape: st.out_shape }].filter(x => x.shape);
    $("stTrack").innerHTML = track.map(x => `<span class="tchip${x.name.length === 1 ? " qkv" : ""}" ${x.stage != null ? `data-stage="${x.stage}"` : ""}><b>${esc(x.name)}</b> ${esc(fmt(x.shape))}${x.note ? ` <i>${esc(x.note)}</i>` : ""}</span>`).join("");
    const all = (S.parts() || []).filter(Boolean);
    // Worked arithmetic goes in a text box below the drawing: readable at any size, never clipped.
    const notes = all.filter(p => p.lines), parts = all.filter(p => !p.lines);
    const tex = l => { try { return window.katex.renderToString(l, { throwOnError: false, displayMode: false }); } catch { return esc(l); } };
    $("stWork").innerHTML = notes.map(n => {
      const useTex = window.katex && (n.tex || true);
      // Notes without explicit LaTeX are typeset as text so every calculation looks the same.
      const rows = useTex ? (n.tex || n.lines.map(l => txt(l))).map(l => `<div class="wl">${tex(l)}</div>`) : n.lines.map(l => `<div>${esc(l)}</div>`);
      return `<div class="wk${n.noteCls ? " " + n.noteCls : ""}${useTex ? " tex" : ""}">${rows.join("")}</div>`;
    }).join("");
    $("stWork").hidden = !notes.length;
    // Pick the arrangement (side by side or stacked) that draws the tensors largest in this panel.
    const box = $("stSvg").parentNode.getBoundingClientRect();
    const fitScale = l => Math.min((box.width || 600) / l.vw, (box.height || 300) / l.vh);
    const row = T.layout(parts, Infinity), col = T.layout(parts, 0);
    const L = fitScale(row) >= fitScale(col) ? row : col;
    const svg = $("stSvg");
    // Don't blow simple views up: cap the zoom so text stays a normal size.
    const MAXS = 1.7, sc = fitScale(L);
    if (sc > MAXS && box.width && box.height) {
      const vw = box.width / MAXS, vh = box.height / MAXS;
      svg.setAttribute("viewBox", `${(-(vw - L.vw) / 2).toFixed(1)} ${(-(vh - L.vh) / 2).toFixed(1)} ${vw.toFixed(1)} ${vh.toFixed(1)}`);
    } else svg.setAttribute("viewBox", `0 0 ${Math.max(L.vw, 1)} ${Math.max(L.vh, 1)}`);
    svg.innerHTML = `<defs><marker id="sarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--accent)"/></marker></defs>${L.body}`;
    $("stEq").innerHTML = model.eq;
    $("stEq").querySelectorAll(".term").forEach(el => el.classList.toggle("on", el.dataset.term === S.term || S.term === "all"));
    $("stHeadSel").innerHTML = DETAIL && model.heads > 1
      ? `head <button type="button" class="pbtn" data-hd="-1">‹</button> <b>${headSel + 1} / ${model.heads}</b> <button type="button" class="pbtn" data-hd="1">›</button>` : "";
    $("stCount").textContent = `${stage + 1} / ${model.stages.length}`;
    $("stDots").innerHTML = model.stages.map((x, i) => `<button type="button" class="sdot${i === stage ? " on" : ""}" data-si="${i}" title="${esc(x.title)}"></button>`).join("");
    $("stStageTitle").textContent = S.title;
    // The code panel uses the same numbering as the stages.
    const sb = S.sub != null && st.subs ? st.subs[S.sub] : null;
    const pn = document.getElementById("phNote");
    if (pn) pn.textContent = `Stage ${stage + 1}/${model.stages.length} · ${S.title}${sb && sb.file ? ` · line ${sb.line} (outlined)` : sb ? " · computed outside the model" : ""}`;
    const nextBtn = root.querySelector('[data-sd="1"]');
    const last = stage >= model.stages.length - 1, lastStep = App.state().cur >= STEPS.length - 1;
    nextBtn.textContent = last && !lastStep ? "next step ›" : "›";
    nextBtn.disabled = last && lastStep;
    $("stHow").textContent = S.how || "";
    $("stFoot").textContent = DETAIL ? `${App.preset.label}: every value is shown. ★ = max · red = negative · mint = zeroed or −∞ · arrows show the direction of the computation.`
      : !hasVals(st) ? `Values are recorded for layer 1; layer ${st.layer} runs the same operation (shapes shown).`
      : "Orange = positive, blue = negative, stronger = larger. Feature axes are sampled (see each axis label); switch Config to Tiny or Small to see every value.";
  }

  function setStage(i, fromApp = false) {
    if (!model) return;
    stage = Math.max(0, Math.min(model.stages.length - 1, i));
    const want = model.stages[stage].sub;
    if (!fromApp) {
      const target = want == null ? -1 : want;
      if (App.state().sub !== target) App.selectSub(target, true);
    }
    draw();
  }

  function rebuild(keepStage = false) {
    const { cur } = App.state();
    const st = STEPS[cur];
    GROUPS.forEach(g => { if (st.key.startsWith(g.which + "_") && st.layer) lastLayer[g.which] = st.layer; });
    if (forStep !== cur) { Object.keys(picks).forEach(k => delete picks[k]); headSel = 0; }
    model = buildFor(st);
    if (!keepStage || forStep !== cur) stage = 0;
    forStep = cur;
    // Tell the code panel which line this stage runs (also for the first stage of a new step).
    const want = model.stages[stage] && model.stages[stage].sub;
    setStage(stage, want == null);
  }

  // ---------------------------------------------------------------- events
  root.addEventListener("click", e => {
    const hd = e.target.closest("[data-hd]");
    if (hd && model.heads) { headSel = (headSel + +hd.dataset.hd + model.heads) % model.heads; return draw(); }
    const pc = e.target.closest(".gcell.pick");
    if (pc) { picks[stage] = { i: +pc.dataset.r, j: +pc.dataset.c }; return draw(); }
    const sd = e.target.closest("[data-sd]");
    if (sd) {
      const dlt = +sd.dataset.sd;
      if (dlt > 0 && model && stage >= model.stages.length - 1) return App.goto(Math.min(STEPS.length - 1, App.state().cur + 1));
      return setStage(stage + dlt);
    }
    const dot = e.target.closest("[data-si]");
    if (dot) return setStage(+dot.dataset.si);
    const tc = e.target.closest(".tchip[data-stage]");
    if (tc) return setStage(+tc.dataset.stage);
    const smp = e.target.closest("[data-sample]");
    if (smp) return App.setSample(+smp.dataset.sample);
    const cl = e.target.closest(".clayer");
    if (cl) {
      const which = cl.dataset.which, g = GROUPS.find(x => x.which === which), N = g.N;
      const nl = ((lastLayer[which] - 1 + +cl.dataset.d + N) % N) + 1;
      const st = STEPS[App.state().cur];
      const key = st.key.startsWith(which + "_") && st.layer ? st.key : g.keys[0];
      lastLayer[which] = nl;
      return App.goto(stepIndex(key, nl));
    }
    const cr = e.target.closest(".crumb");
    if (cr) return App.goto(stepIndex(cr.dataset.key, cr.dataset.layer ? +cr.dataset.layer : null));
  });

  App.on("step", () => rebuild());
  App.on("sub", ({ sub }) => {                       // e.g. keyboard: follow the code panel's sub-step
    if (!model) return;
    const cur = model.stages[stage] && model.stages[stage].sub;
    if ((cur == null ? -1 : cur) === sub) return;
    const i = model.stages.findIndex(s => (s.sub == null ? -1 : s.sub) === sub);
    if (i >= 0) { stage = i; draw(); }
  });
  App.on("sample", () => rebuild(true));
  App.on("resize", id => { if (id === "stage") draw(); });

  // Playback driver: the main Play walks through the stages before moving to the next step.
  let dwell = 0;   // extra playback ticks to stay on the current stage
  App.stageDriver = {
    advance() {
      if (!model) return false;
      if (dwell > 0) { dwell--; return true; }
      if (stage >= model.stages.length - 1) return false;
      setStage(stage + 1);
      dwell = model.stages[stage].pick ? 2 : 0;          // matmul / arithmetic stages stay ~3× longer
      return true;
    },
    step(dlt) { setStage(stage + dlt); },
  };

  rebuild();
  // ?stage=<n> opens that stage of the initial step (used by demo/capture.py).
  const startStage = +new URLSearchParams(location.search).get("stage");
  if (startStage > 1) setStage(startStage - 1);
})();
