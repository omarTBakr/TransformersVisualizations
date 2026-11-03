/*
 * Encoder–decoder steps for the shared operation stage (shared/ui/js/stage-core.js):
 * token embeddings, positional encoding, the padding / causal masks, the generator, and the
 * two-row breadcrumb. Attention, LayerNorm, feed-forward and softmax come from the core.
 */
window.StageModel = function (api) {
  "use strict";
  const { App, T, cfg, B, STEPS, d, h, esc, fmt, word, sample, subIdx, subBy, byLabel, byVar, words, nf,
          cubeOf, subPart, shapeOnly, eqTerm, mat, matOf, transpose, G, rowMaxFlag, txt, t3, ts3, n3, s3,
          matmul, pickOf } = api;

  // Positional-encoding values for the sampled dims, from the paper's formula.
  function peSlice(xs) {
    const v = xs.v.map(seq => seq.map((_, pos) => xs.dims.map(dim => {
      const i2 = dim - (dim % 2);
      const ang = pos / Math.pow(10000, i2 / d);
      return dim % 2 ? Math.cos(ang) : Math.sin(ang);
    })));
    return { v, D: xs.D, dims: xs.dims };
  }

  // ---------------------------------------------------------------- stage builders
  function embeddingStages(st) {
    const side = st.key === "input_embedding" ? "src" : "tgt";
    const look = subIdx(st, byLabel(/^Embedding lookup/)), scale = subIdx(st, byLabel(/^Scale by/));
    const lookV = st.subs[look] && st.subs[look].vals;
    const stages = [{
      title: "Token ids", term: "tok", sub: null,
      how: "The tokenizer turned each word into an id (a toy word-level vocabulary here).",
      parts: () => [T.tokens(side)],
    }];
    words(side).forEach((w, t) => stages.push({
      title: `Look up “${w}”`, term: "e", sub: look,
      how: `Row ${T.idsOf(side)[sample()][t]} of the ${side === "src" ? cfg.src_vocab_size : cfg.tgt_vocab_size} × ${d} table becomes the vector for position ${t}.`,
      parts: () => [lookV ? cubeOf({ v: lookV.v, D: lookV.full[2] }, { single: true, hiCols: new Set([t]), title: "E[token]" }) : shapeOnly(st.out_shape, "lookup")],
    }));
    stages.push({
      title: `Scale by √${d}`, term: "s", sub: scale,
      how: `Every value is multiplied by √${d} ≈ ${Math.sqrt(d).toFixed(1)}, so embeddings and positional encodings have similar sizes.`,
      parts: () => [subPart(st, look, { single: true, title: "E[token]" }), T.op(`× ${Math.sqrt(d).toFixed(1)}`, `√${d}`), cubeOf(st.out_vals, { single: true, title: "X" })],
    });
    return {
      title: `${side === "src" ? "Input" : "Output"} Embedding`,
      eq: `X = ${eqTerm("e", "E[token]")} · ${eqTerm("s", `√d<sub>model</sub>`)} ${eqTerm("tok", "")}`,
      stages,
    };
  }

  function peStages(st) {
    const add = subIdx(st, byLabel(/positional encoding/));
    const X = st.in_vals, Y = st.out_vals;
    const P = X ? peSlice(X) : null;
    const side = X ? T.sideOfT(X.v[0].length) : "src";
    const stages = [{
      title: "X + PE = Y", term: "all", sub: add,
      how: "PE doesn't depend on the words, only on the position, so the same table is added to every sample.",
      parts: () => [cubeOf(X, { single: true, title: "X (embeddings)", cw: 20 }), T.op("+"), cubeOf(P, { single: true, title: "PE (sin / cos)", cw: 20 }),
                    T.op("="), cubeOf(Y, { single: true, title: "Y", cw: 20 })],
    }];
    if (X && Y) words(side).forEach((w, t) => stages.push({
      title: `Position ${t}: “${w}”`, term: "pe", sub: add,
      how: `Column ${t} of PE is added to column ${t} of X. Even dims use sin, odd dims use cos.`,
      parts: () => {
        const s = sample(), hi = new Set([t]);
        const cell = k => `dim ${X.dims[k]} (${X.dims[k] % 2 ? "cos" : "sin"}): ${nf(X.v[s][t][k])} + ${nf(P.v[s][t][k])} = ${nf(Y.v[s][t][k])}`;
        return [cubeOf(X, { single: true, hiCols: hi, title: "X", cw: 20 }), T.op("+"), cubeOf(P, { single: true, hiCols: hi, title: "PE", cw: 20 }),
                T.op("="), cubeOf(Y, { single: true, hiCols: hi, title: "Y", cw: 20 }), T.note([cell(0), cell(1)], { w: 290 })];
      },
    }));
    stages.push({
      title: "Y feeds Q, K and V", term: "y", sub: add,
      how: "The next block projects this same tensor three times: queries, keys and values all start from Y.",
      parts: () => [cubeOf(Y, { title: "Y" }), T.op("→"), T.note(["Q = Y·W_Q", "K = Y·W_K", "V = Y·W_V"], { w: 140 })],
    });
    return {
      title: "Positional Encoding",
      eq: `${eqTerm("y", "Y")} = ${eqTerm("x", "X")} + ${eqTerm("pe", "PE")}<br>
           PE<sub>(pos, 2i)</sub> = sin(pos / 10000<sup>2i/d</sup>) &nbsp;·&nbsp; PE<sub>(pos, 2i+1)</sub> = cos(pos / 10000<sup>2i/d</sup>)
           ${eqTerm("all", "")}`,
      stages,
    };
  }

  function maskStages(st) {
    const sub = (st.subs || []).length ? 0 : null;
    if (st.key === "causal_mask") {
      const lab = words("tgt");
      return { title: "Causal mask", eq: `M<sub>ij</sub> = ${eqTerm("m", "[ j > i ]")}`,
        stages: [{ title: "Block the future", term: "m", sub, how: "Query i may only look at keys 0…i; everything above the diagonal is blocked.",
          parts: () => [T.valueGrid(st.mask, r => lab[r], c => lab[c], "bool", { title: `sample ${sample() + 1}` })] }] };
    }
    const side = st.key === "src_pad_mask" ? "src" : "tgt";
    return { title: "Padding mask", eq: `M = ${eqTerm("m", "(tokens == pad)")}`,
      stages: [{ title: "Find the padding", term: "m", sub, how: "True marks padding; attention will give those keys zero weight.",
        parts: () => [T.tokens(side), T.op("== pad"), T.valueGrid(st.mask, r => `sample ${r + 1}`, c => String(c), "bool")] }] };
  }

  function generatorStages(st) {
    const tk = st.topk_logits;
    return { title: "Linear (generator)", eq: `${eqTerm("l", "logits = x W<sup>T</sup>")}  (W tied to the output embedding)`,
      stages: [{ title: `Project ${d} → ${cfg.tgt_vocab_size}`, term: "l", sub: null,
        how: "One score per vocabulary word, for every position.",
        parts: () => [cubeOf(st.in_vals, { single: true, title: "x", cw: 18 }), T.op("× Wᵀ"), cubeOf(st.out_vals, { single: true, title: "logits (16 of 32000 shown)", cw: 18 }),
                      tk ? T.note(["top logits, last word:", ...tk.ids.slice(0, 3).map((id, i) => `${B.tgt_vocab[id] ?? "token " + id}: ${tk.logits[i].toFixed(2)}`)], { w: 190 }) : null].filter(Boolean) }] };
  }

  function embeddingD(st) {
    const side = st.key === "input_embedding" ? "src" : "tgt";
    const look = subIdx(st, byLabel(/^Embedding lookup/)), scale = subIdx(st, byLabel(/^Scale by/));
    const L = st.subs[look], ids = T.idsOf(side)[sample()];
    const rowIds = L && L.ops ? Object.keys(L.ops.rows).map(Number) : [];
    const table = rowIds.length ? matOf(rowIds.map(id => L.ops.rows[id]), "rows", "features") : null;
    if (table) table.rowLab = r => `${word(rowIds[r], side)} (${rowIds[r]})`;
    const lookM = mat(L), outM = mat({ vals: { v: st.out_vals.v.map(x => x), names: ["batch", "seq", "features"], full: [0, st.out_vals.v[0].length, d], kind: "float" }, shape: [0, 0, d] });
    const stages = [{ title: "Token ids", term: "tok", sub: null, how: "Each word has an id in the vocabulary.", parts: () => [T.tokens(side)] }];
    ids.forEach((id, t) => stages.push({
      title: `Look up “${word(id, side)}” (id ${id})`, term: "e", sub: look,
      how: `Copy row ${id} of the embedding table E into position ${t}.`,
      parts: () => [G(table, { title: "E (rows used)", hiRow: rowIds.indexOf(id), rowArrow: true }), T.op("→", "copy row"), G(lookM, { title: "E[token]", hiRow: t })],
    }));
    stages.push({
      title: `Scale by √${d}`, term: "s", sub: scale, pick: true,
      how: `Every value is multiplied by √${d} ≈ ${Math.sqrt(d).toFixed(3)}. Click a cell to check it.`,
      parts: () => {
        const { i, j } = pickOf({ i: 0, j: 0 });
        const a = lookM.m[i][j], y = outM.m[i][j];
        return [G(lookM, { title: "E[token]", hiCell: { i, j } }), T.op(`× ${Math.sqrt(d).toFixed(3)}`, `√${d}`), G(outM, { title: "X (click a cell)", hiCell: { i, j }, gridId: "C" }),
                T.note([`${n3(a)} × √${d} = ${n3(a)} × ${Math.sqrt(d).toFixed(3)} = ${n3(a * Math.sqrt(d))}   (model: ${n3(y)})`],
                  { tex: [`x_{${i},${j}} = E[\\text{token}]_{${j}}\\times\\sqrt{${d}} = ${t3(a)}\\times ${Math.sqrt(d).toFixed(3)} = ${t3(a * Math.sqrt(d))}\\quad ${txt(`(model: ${n3(y)})`)}`] })];
      },
    });
    return { ...embeddingStages(st), stages };
  }

  function peD(st) {
    const add = subIdx(st, byLabel(/positional encoding/));
    const X = st.in_vals, Y = st.out_vals, P = peSlice(X), b = sample();
    const Xm = matOf(X.v[b], "seq", "features"), Pm = matOf(P.v[b], "seq", "features"), Ym = matOf(Y.v[b], "seq", "features");
    const stages = [{ title: "X + PE = Y", term: "all", sub: add, how: "PE depends only on the position, so the same table is added to every sample.",
      parts: () => [G(Xm, { title: "X" }), T.op("+"), G(Pm, { title: "PE" }), T.op("="), G(Ym, { title: "Y" })] }];
    Xm.m.forEach((_, t) => stages.push({ title: `Position ${t}: “${Xm.rowLab(t)}”`, term: "pe", sub: add,
      how: "Row t of PE is added to row t of X, element by element. Even dims use sin, odd dims cos.",
      parts: () => [G(Xm, { title: "X", hiRow: t, rowArrow: true }), T.op("+"), G(Pm, { title: "PE", hiRow: t, rowArrow: true }), T.op("="), G(Ym, { title: "Y", hiRow: t }),
                    T.note(Xm.m[t].slice(0, 9).map((x, j) => { const i2 = j - (j % 2);
                      return `f${j}: ${n3(x)} + ${j % 2 ? "cos" : "sin"}(${t} / 10000^(${i2}/${d})) = ${n3(x)} + ${s3(Pm.m[t][j])} = ${n3(Ym.m[t][j])}`; }),
                      { tex: Xm.m[t].slice(0, 9).map((x, j) => { const i2 = j - (j % 2);
                        return `y_{${t},${j}} = x_{${t},${j}} + \\${j % 2 ? "cos" : "sin"}\\!\\left(\\frac{${t}}{10000^{${i2}/${d}}}\\right) = ${t3(x)} + ${ts3(Pm.m[t][j])} = ${t3(Ym.m[t][j])}`; }) })] }));
    stages.push({ title: "Y feeds Q, K and V", term: "y", sub: add, how: "The next block projects this same tensor into queries, keys and values.",
      parts: () => [G(Ym, { title: "Y" }), T.op("→"), T.note(["Q = Y·W_Q", "K = Y·W_K", "V = Y·W_V"], { w: 140 })] });
    return { ...peStages(st), stages };
  }

  function generatorD(st) {
    const base = generatorStages(st);
    if (!st.ops) return base;
    const b = sample();
    return { ...base, stages: [{ title: `logits = x · Wᵀ (${d} → ${cfg.tgt_vocab_size} words)`, term: "l", sub: null, pick: true,
      how: "One score per vocabulary word. W is tied to the output embedding. Click a cell.",
      parts: () => matmul({ A: matOf(st.ops.x[b], "seq", "features"), Bm: matOf(transpose(st.ops.W), "features", "vocab"),
                            Cm: matOf(st.out_vals.v[b], "seq", "vocab"), tA: "x", tB: "Wᵀ", tC: "logit" }).parts() }] };
  }

  function build(st) {
    const D = api.DETAIL;
    if (/embedding$/.test(st.key)) return D ? embeddingD(st) : embeddingStages(st);
    if (/^pe_/.test(st.key)) return D ? peD(st) : peStages(st);
    if (/mask$/.test(st.key)) return maskStages(st);
    if (st.key === "generator") return D ? generatorD(st) : generatorStages(st);
    return null;
  }

  function parallel(st) {
    const Bn = B.src.length, Tn = st.out_shape && st.out_shape.length > 1 ? st.out_shape[1] : 0;
    if (/embedding$/.test(st.key)) return `One indexing operation gathers all ${Bn} × ${Tn} = ${Bn * Tn} embedding rows at once; the word-by-word stages are only for explanation.`;
    if (/^pe_/.test(st.key)) return `x + pe[:, :T] adds the whole (1, ${Tn}, ${d}) table to all ${Bn} samples in one broadcast addition; there is no loop over positions in the code.`;
    if (/mask$/.test(st.key)) return "One element-wise comparison over the whole tensor builds the entire mask.";
    if (st.key === "generator") return `One matrix multiplication scores every position against the whole vocabulary: (${Bn}·${Tn}, ${d}) × (${d}, ${cfg.tgt_vocab_size}).`;
    return null;
  }

  const ENC = ["enc_self_attn", "enc_norm1", "enc_ffn", "enc_norm2"];
  const DEC = ["dec_masked_attn", "dec_norm1", "dec_cross_attn", "dec_norm2", "dec_ffn", "dec_norm3"];
  const crumbs = {
    short: { input_embedding: "Embedding", pe_enc: "PE", enc_self_attn: "MHA", enc_norm1: "A&N", enc_ffn: "FFN", enc_norm2: "A&N",
      enc_final: "Final LN", output_embedding: "Embedding", pe_dec: "PE", causal_mask: "Causal mask", src_pad_mask: "Pad mask",
      tgt_pad_mask: "Pad mask", dec_masked_attn: "Masked MHA", dec_norm1: "A&N", dec_cross_attn: "Cross", dec_norm2: "A&N",
      dec_ffn: "FFN", dec_norm3: "A&N", dec_final: "Final LN", generator: "Linear", softmax: "Softmax" },
    rows: [
      { label: "encoder", items: ["src_pad_mask", "input_embedding", "pe_enc", { group: "Enc", which: "enc", keys: ENC, N: cfg.n_encoder_layers }, "enc_final"], tail: "→ memory" },
      { label: "decoder", items: ["tgt_pad_mask", "causal_mask", "output_embedding", "pe_dec", { group: "Dec", which: "dec", keys: DEC, N: cfg.n_decoder_layers },
                                  "dec_final", "generator", "softmax"] },
    ],
  };

  return { build, parallel, crumbs };
};
