/*
 * Isometric tensor drawing helpers (pure functions → SVG fragments {w, h, body}).
 *
 *   const T = Tensors.create({ B, cfg, word, esc, sample: () => 0 });
 *   const part = T.cube({ v, ... });            // 3-D: v[depth][x][y]
 *   const svg  = T.layout([partA, T.op("+"), partB]);
 *
 * Colours come from CSS tokens (--pos / --neg / --mint), so light and dark themes work.
 */
(function () {
  "use strict";

  function create(ctx) {
    const { B, word, esc } = ctx;
    const sample = () => ctx.sample();
    // Encoder-only models (ViT) have a single sequence: `tgt` is absent.
    const SRC_T = B.src[0].length, TGT_T = B.tgt ? B.tgt[0].length : -1;
    const sideOfT = T => (T === SRC_T || !B.tgt ? "src" : "tgt");
    const idsOf = side => (side === "src" || !B.tgt ? B.src : B.tgt);
    const short = w => (w.length > 6 ? w.slice(0, 5) + "…" : w);
    const NEG_INF = 1e29;

    // Measure text with the same fonts the stylesheet uses, so boxes can be sized to fit their text.
    const SANS = 'system-ui, Cantarell, "Noto Sans", "DejaVu Sans", Arial, sans-serif';
    const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
    const meas = document.createElement("canvas").getContext("2d");
    function textW(str, px, weight = 400, mono = false) {
      meas.font = `${weight} ${px}px ${mono ? MONO : SANS}`;
      return meas.measureText(String(str)).width;
    }
    const maxW = (arr, px, weight, mono) => arr.reduce((m, t) => Math.max(m, textW(t, px, weight, mono)), 0);

    const heat = (v, max) => {
      const a = (0.08 + 0.87 * Math.min(1, Math.abs(v) / (max || 1))).toFixed(3);
      return `rgba(var(${v >= 0 ? "--pos" : "--neg"}), ${a})`;
    };
    const fmtNum = x => (Math.abs(x) >= NEG_INF ? "−∞" : Math.abs(x) >= 100 || (Math.abs(x) < 0.01 && x) ? x.toExponential(1) : x.toFixed(2));

    // Labels / padding flags for one axis, from its name and length.
    function axisLabels(name, n) {
      if (name === "batch") return i => `sample ${i + 1}`;
      if (name === "heads") return i => `head ${i + 1}`;
      if (["seq", "queries", "keys"].includes(name) && (n === SRC_T || n === TGT_T)) {
        const side = sideOfT(n);
        return i => word(idsOf(side)[sample()][i], side);
      }
      return i => String(i);
    }
    function axisPad(name, n) {
      if (!["seq", "queries", "keys"].includes(name) || (n !== SRC_T && n !== TGT_T)) return () => false;
      const ids = idsOf(sideOfT(n))[sample()];
      return i => ids[i] === B.pad;
    }

    /*
     * Cube: v[depth][x][y] drawn as slabs; `front` slab on top with full cells.
     * Options: generic (no values, grey cells), hiCols (Set of x to outline), hiCell {t, k},
     * heads (draw head bands), dimmed (shape only), title (caption above).
     */
    function cube({ v, front = 0, depthLabels = i => String(i), xLabels = i => String(i), xPad = () => false,
                    xName = "", yName = "", depthName = "", heads = 0, generic = false, dimmed = false,
                    hiCols = null, hiCell = null, title = "", cw: cwIn = 26, numbers = false }) {
      const Dn = v.length, T = v[0].length, K = v[0][0].length;
      // Cells must fit the column labels (and the numbers, when shown).
      const labW = generic ? 0 : maxW([...Array(T).keys()].map(t => short(String(xLabels(t)))), 9.5, 600);
      const numW = numbers ? maxW(v[front].flat().map(fmtNum), 8, 400, true) : 0;
      const cw = Math.ceil(Math.max(numbers ? 32 : cwIn, labW + 4, numW + 6));
      const ch = numbers ? 15 : Math.max(5, Math.min(9, Math.floor(130 / K))), sx = 12, sy = -9;
      const gx = Dn > 2 ? Math.max(10, Math.floor(110 / (Dn - 1))) : 34, gy = -Math.round(gx * 0.76);
      const flat = v.flat(2).filter(x => Math.abs(x) < NEG_INF);
      const max = Math.max(1e-12, ...flat.map(Math.abs));
      const fill = x => (generic || dimmed ? "var(--surface-2)" : Math.abs(x) >= NEG_INF ? "var(--mint)" : heat(x, max));
      const W = T * cw, H = K * ch, ox = 38, oy = 22 + (title ? 14 : 0) + (Dn - 1) * -gy - sy;
      const order = [...Array(Dn).keys()].filter(d => d !== front).reverse().concat([front]);
      let body = title ? `<text class="ptitle" x="${ox}" y="12">${esc(title)}</text>` : "";
      order.forEach((d, j) => {
        const back = d === front ? 0 : (Dn - 1 - j);
        const x0 = ox + back * gx, y0 = oy + back * gy;
        let g = `<polygon class="solid" points="${x0},${y0} ${x0 + sx},${y0 + sy} ${x0 + W + sx},${y0 + sy} ${x0 + W + sx},${y0 + H + sy} ${x0 + W},${y0 + H} ${x0},${y0 + H}"/>`;
        for (let t = 0; t < T; t++) g += `<polygon class="face" points="${x0 + t * cw},${y0} ${x0 + (t + 1) * cw},${y0} ${x0 + (t + 1) * cw + sx},${y0 + sy} ${x0 + t * cw + sx},${y0 + sy}" style="fill:${fill(v[d][t][0])}"/>`;
        for (let k = 0; k < K; k++) g += `<polygon class="face side" points="${x0 + W},${y0 + k * ch} ${x0 + W + sx},${y0 + k * ch + sy} ${x0 + W + sx},${y0 + (k + 1) * ch + sy} ${x0 + W},${y0 + (k + 1) * ch}" style="fill:${fill(v[d][T - 1][k])}"/>`;
        if (back === 0) {
          const showNums = numbers && !generic && !dimmed;
          for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) {
            const x = v[d][t][k];
            if (showNums) {
              g += `<rect class="cell" x="${x0 + t * cw}" y="${y0 + k * ch}" width="${cw}" height="${ch}" style="fill:${fill(x)}"><title>${esc(`${depthLabels(d)}, ${xLabels(t)}, ${yName} ${k}: ${fmtNum(x)}`)}</title></rect>
                <text class="cnum" x="${x0 + t * cw + cw / 2}" y="${y0 + k * ch + ch / 2 + 3}" text-anchor="middle">${fmtNum(x)}</text>`;
              continue;
            }
            const tip = generic ? `${depthName} × ${xName} × ${yName}` : `${depthLabels(d)}, ${xLabels(t)}, ${yName} ${k}: ${fmtNum(x)}`;
            g += `<rect class="cell" x="${x0 + t * cw}" y="${y0 + k * ch}" width="${cw}" height="${ch}" style="fill:${fill(x)}"><title>${esc(tip)}</title></rect>`;
          }
          for (let t = 0; t < T; t++) if (xPad(t)) g += `<rect class="padcol" x="${x0 + t * cw + 1}" y="${y0 + 1}" width="${cw - 2}" height="${H - 2}"/>`;
          if (heads && K % heads === 0 && K > heads)
            for (let hh = 1; hh < heads; hh++) g += `<line class="headline" x1="${x0}" x2="${x0 + W}" y1="${y0 + hh * (K / heads) * ch}" y2="${y0 + hh * (K / heads) * ch}"/>`;
          // Highlight a column as a 3-D slab that follows the isometric projection:
          // tinted front + top faces, then the outline of the slab and its two inner edges.
          if (hiCols) hiCols.forEach(t => {
            const xl = x0 + t * cw, xr = xl + cw;
            const pts = a => a.map(p => p.join(",")).join(" ");
            g += `<g class="hislab">
              <rect class="hifill" x="${xl}" y="${y0}" width="${cw}" height="${H}"/>
              <polygon class="hifill top" points="${pts([[xl, y0], [xr, y0], [xr + sx, y0 + sy], [xl + sx, y0 + sy]])}"/>
              <polygon class="hiedge" points="${pts([[xl, y0 + H], [xl, y0], [xl + sx, y0 + sy], [xr + sx, y0 + sy], [xr + sx, y0 + sy + H], [xr, y0 + H]])}"/>
              <polyline class="hiedge" points="${pts([[xl, y0], [xr, y0], [xr, y0 + H]])}"/>
              <line class="hiedge" x1="${xr}" y1="${y0}" x2="${xr + sx}" y2="${y0 + sy}"/></g>`;
          });
          if (hiCell) g += `<rect class="hicell" x="${x0 + hiCell.t * cw}" y="${y0 + hiCell.k * ch}" width="${cw}" height="${ch}"/>`;
          for (let t = 0; t < T; t++) {
            const lab = xLabels(t);
            g += `<text class="tw${xPad(t) ? " pad" : ""}${hiCols && hiCols.has(t) ? " hi" : ""}" x="${x0 + t * cw + cw / 2}" y="${y0 + H + 12}" text-anchor="middle">${esc(short(lab))}</text>`;
          }
          g += `<text class="ax" x="${x0 + W / 2}" y="${y0 + H + 27}" text-anchor="middle">${esc(xName)}</text>`;
          const yLen = textW(yName, 9.5), room = H + 24;
          g += `<text class="ax" transform="translate(${x0 - 8},${y0 + H / 2}) rotate(-90)" text-anchor="middle"${yLen > room ? ` textLength="${room}" lengthAdjust="spacingAndGlyphs"` : ""}>${esc(yName)}</text>`;
          if (Dn > 1) g += `<text class="ax hd" x="${x0}" y="${y0 + sy - 3}">${esc(depthLabels(d))} (front)</text>`;
        } else {
          g += `<rect class="cell" x="${x0}" y="${y0}" width="${W}" height="${H}" style="fill:var(--surface-2)"/>`;
        }
        body += `<g class="${back ? "slab back" : "slab"}">${g}</g>`;
      });
      if (Dn > 1 && depthName) body += `<text class="ax" x="${ox + W + sx + gx * (Dn - 1) + 4}" y="${oy + sy + gy * (Dn - 1) + 10}">${esc(depthName)}</text>`;
      const right = ox + W + sx + gx * (Dn - 1) + (depthName ? textW(depthName, 9.5) + 12 : 30);
      const titleR = title ? ox + textW(title, 11, 700) + 6 : 0;
      const axisR = ox + Math.max(textW(xName, 9.5), 0) + 6;
      return { w: Math.ceil(Math.max(right, titleR, axisR)), h: oy + H + 34, body };
    }

    // Long sequences (ViT-Base: 197 tokens) are cut to their first positions; slice.T is the full length.
    const seqName = (shown, full) => (full && full > shown ? `sequence · first ${shown} of T = ${full}` : `sequence · T = ${shown}`);

    // A cube from a simulator slice {v: [B][T][K], D, dims} (batch in depth, selected sample in front).
    function batchCube(slice, opts = {}) {
      const T = slice.v[0].length, side = sideOfT(T);
      return cube({
        v: slice.v, front: sample(), depthLabels: i => `sample ${i + 1}`, depthName: `batch · B = ${slice.v.length}`,
        xLabels: i => word(idsOf(side)[sample()][i], side), xPad: i => idsOf(side)[sample()][i] === B.pad,
        xName: seqName(T, slice.T), yName: `features · ${slice.D}${opts.heads ? ` · ${opts.heads} heads` : ""}`, ...opts,
      });
    }

    // A generic (value-free) cube with named axes, for explaining an operation's geometry.
    function genericCube({ T = 6, K = 10, Dn = 2, xName, yName, depthName, hiCols = null, title = "" }) {
      const v = Array.from({ length: Dn }, () => Array.from({ length: T }, () => Array(K).fill(0)));
      return cube({ v, generic: true, xName, yName, depthName, hiCols, title,
                    depthLabels: i => `b${i + 1}`, xLabels: i => `t${i + 1}` });
    }

    /*
     * 2-D grid (token ids, masks, attention maps, matrices).
     * opts: rowLabel, colLabels, title, hiRow, hiCol, hiCell {i, j}, rowArrow, colArrow,
     *       gridId (cells get data-grid / data-r / data-c so they can be clicked), cw.
     */
    function grid(rows, cols, cell, { rowLabel = r => r, colLabels = null, hiRow = null, hiCol = null, hiCell = null,
                                      rowArrow = false, colArrow = false, gridId = null, title = "", cw = 30 } = {}) {
      const ch = 20, sx = 6, sy = -5;
      // Pre-compute the cells so column width, row-label gutter and total width fit every text.
      const cells = rows.map((_, i) => cols.map((_, j) => cell(i, j)));
      const cellW = cells.flat().reduce((m, c) => Math.max(m, textW(c.text, /\bnum\b/.test(c.cls) ? 8.5 : 9.5, /\bnum\b/.test(c.cls) ? 500 : 600)), 0);
      const colW = colLabels ? maxW(colLabels.map(c => short(String(c))), 9.5, 400) : 0;
      cw = Math.ceil(Math.max(cw, cellW + 8, colW + 4));
      const lw = Math.ceil(Math.max(30, maxW(rows.map((r, i) => rowLabel(r, i)), 9.5, 700) + 12));
      const top = 10 + (colLabels ? 14 : 0) + (title ? 14 : 0) - sy;
      let body = title ? `<text class="ptitle" x="${lw}" y="12">${esc(title)}</text>` : "";
      rows.forEach((r, i) => {
        body += `<text class="ax${i === hiRow ? " hi" : ""}" x="${lw - 6}" y="${top + i * ch + 14}" text-anchor="end">${esc(rowLabel(r, i))}</text>`;
        cols.forEach((c, j) => {
          const { text, cls, tip, style } = cells[i][j];
          const x = lw + j * cw, y = top + i * ch;
          const attrs = gridId ? ` data-grid="${gridId}" data-r="${i}" data-c="${j}"` : "";
          body += `<g class="gcell ${cls}${gridId ? " pick" : ""}"${attrs}><rect x="${x}" y="${y}" width="${cw}" height="${ch}"${style ? ` style="${style}"` : ""}><title>${esc(tip)}</title></rect>
            <text x="${x + cw / 2}" y="${y + 14}" text-anchor="middle">${esc(text)}</text></g>`;
        });
      });
      const W = cols.length * cw, H = rows.length * ch, x0 = lw, y0 = top;
      body = `<polygon class="face" points="${x0},${y0} ${x0 + W},${y0} ${x0 + W + sx},${y0 + sy} ${x0 + sx},${y0 + sy}"/>
              <polygon class="face side" points="${x0 + W},${y0} ${x0 + W + sx},${y0 + sy} ${x0 + W + sx},${y0 + H + sy} ${x0 + W},${y0 + H}"/>` + body;
      if (hiRow != null) body += `<rect class="hicol" x="${x0 - 2}" y="${y0 + hiRow * ch}" width="${W + 4}" height="${ch}" rx="2"/>`;
      if (hiCol != null) body += `<rect class="hicol" x="${x0 + hiCol * cw}" y="${y0 - 2}" width="${cw}" height="${H + 4}" rx="2"/>`;
      if (hiCell) body += `<rect class="hicell big" x="${x0 + hiCell.j * cw}" y="${y0 + hiCell.i * ch}" width="${cw}" height="${ch}" rx="2"/>`;
      // Direction of the computation: along a row (→) and down a column (↓).
      // Arrows sit just outside the grid so they never cover a number.
      if (rowArrow && hiRow != null) body += `<line class="flowline" x1="${x0 + W + sx + 1}" y1="${y0 + hiRow * ch + ch / 2}" x2="${x0 + W + sx + 14}" y2="${y0 + hiRow * ch + ch / 2}" marker-end="url(#sarr)"/>`;
      if (colArrow && hiCol != null) body += `<line class="flowline" x1="${x0 + hiCol * cw + cw / 2}" y1="${y0 + H + 1}" x2="${x0 + hiCol * cw + cw / 2}" y2="${y0 + H + 14}" marker-end="url(#sarr)"/>`;
      if (colLabels) colLabels.forEach((c, j) => body += `<text class="ax${j === hiCol ? " hi" : ""}" x="${lw + j * cw + cw / 2}" y="${top + sy - 4}" text-anchor="middle">${esc(short(c))}</text>`);
      const titleR = title ? lw + textW(title, 11, 700) + 6 : 0;
      return { w: Math.ceil(Math.max(lw + W + sx + 22, titleR)), h: top + H + 20, body };
    }

    function valueGrid(v, rowLab, colLab, kind, opts = {}) {
      const flat = v.flat().filter(x => Math.abs(x) < NEG_INF);
      const max = Math.max(1e-12, ...flat.map(Math.abs));
      const flag = opts.flags || (() => "");
      return grid(v, v[0], (i, j) => {
        const x = v[i][j], f = flag(i, j, x);
        const extra = f ? ` flag-${f}` : "";
        if (kind === "bool") return { text: x ? "T" : "F", cls: (x ? "blk" : "") + extra, tip: `${rowLab(i)}, ${colLab(j)}: ${x ? "True" : "False"}` };
        if (Math.abs(x) >= NEG_INF) return { text: "−∞", cls: "blk" + extra, tip: `${rowLab(i)}, ${colLab(j)}: −∞ (blocked)` };
        return { text: (f === "max" ? "★" : "") + fmtNum(x), cls: "num" + extra, tip: `${rowLab(i)}, ${colLab(j)}: ${x}${f ? ` (${f})` : ""}`, style: `fill:${heat(x, max)}` };
      }, { rowLabel: (_, i) => rowLab(i), colLabels: v[0].map((_, j) => colLab(j)), ...opts });
    }

    function tokens(side, opts = {}) {
      const ids = idsOf(side);
      return grid(ids, ids[0], (i, j) => {
        const id = ids[i][j];
        return { text: short(word(id, side)), cls: id === B.pad ? "pad" : (id === B.bos || id === B.eos) ? "sp" : "", tip: `${word(id, side)} → id ${id}` };
      }, { rowLabel: (_, i) => `sample ${i + 1}`, ...opts });
    }

    // Any simulator sub-step slice {v, names, full, idx, kind} → the best-fitting drawing.
    function fromSlice(sv, opts = {}) {
      let { v, names, full, idx, kind } = sv;
      const yTitle = (nm, n, sampled) => `${nm} · ${n}${sampled ? ` (${sampled} shown)` : ""}`;
      if (names.length === 4) { v = v[sample()]; names = names.slice(1); full = full.slice(1); }
      if (names.length === 3) {
        const isBatch = names[0] === "batch";
        return cube({
          v, front: isBatch ? sample() : 0,
          depthLabels: axisLabels(names[0], full[0]), depthName: `${names[0]} · ${full[0]}`,
          xLabels: axisLabels(names[1], full[1]), xPad: axisPad(names[1], full[1]),
          xName: `${names[1]} · ${full[1]}`, yName: yTitle(names[2], full[2], idx && idx.length), ...opts,
        });
      }
      if (names.length === 2) return valueGrid(v, axisLabels(names[0], full[0]), axisLabels(names[1], full[1]), kind, opts);
      if (names.length === 1) return valueGrid([v], () => "", axisLabels(names[0], full[0]), kind, opts);
      return null;
    }

    // Operator glyph between operands: big symbol plus a small caption.
    function op(symbol, caption = "") {
      const w = Math.ceil(Math.max(54, textW(symbol, 24, 700) + 16, textW(caption, 10) + 12));
      return { op: true, symbol, caption, w, h: 60, body: `<text class="opsym" x="${w / 2}" y="30" text-anchor="middle">${esc(symbol)}</text>
        <text class="opcap" x="${w / 2}" y="48" text-anchor="middle">${esc(caption)}</text>` };
    }
    // A plain text box (for a value like μ = 0.52).
    function note(lines, { w = 150, cls = "", tex = null } = {}) {
      w = Math.ceil(Math.max(w, maxW(lines, 11.5, 400, true) + 24));     // never narrower than the longest line
      const h = 16 + lines.length * 16;
      return { lines, tex, noteCls: cls, w, h, body: `<rect class="notebox ${cls}" x="2" y="2" width="${w - 4}" height="${h - 4}" rx="8"/>${
        lines.map((l, i) => `<text class="notetext" x="${w / 2}" y="${20 + i * 16}" text-anchor="middle">${esc(l)}</text>`).join("")}` };
    }

    // Lay parts out left-to-right (or stacked if too wide) → {vw, vh, svg markup}.
    function layout(parts, maxW = 620) {
      const gap = 8, rowW = parts.reduce((a, p) => a + p.w + gap, -gap);
      let body = "", vw, vh;
      if (rowW <= maxW) {
        vh = Math.max(40, ...parts.map(p => p.h));
        let x = 0;
        parts.forEach(p => { body += `<g transform="translate(${x},${(vh - p.h) / 2})">${p.body}</g>`; x += p.w + gap; });
        vw = Math.max(rowW, 1);
      } else {
        vw = Math.max(...parts.map(p => (p.op ? 120 : p.w)));
        let y = 0;
        parts.forEach(p => {
          if (p.op) {
            body += `<g transform="translate(${vw / 2 - 60},${y})"><text class="opsym" x="60" y="26" text-anchor="middle">${esc(p.symbol)}</text>
              <text class="opcap" x="60" y="42" text-anchor="middle">${esc(p.caption)}</text></g>`;
            y += 50;
          } else { body += `<g transform="translate(${(vw - p.w) / 2},${y})">${p.body}</g>`; y += p.h + gap; }
        });
        vh = y - gap;
      }
      return { vw, vh, body };
    }

    return { heat, fmtNum, cube, batchCube, genericCube, grid, valueGrid, tokens, fromSlice, op, note, layout,
             axisLabels, axisPad, sideOfT, idsOf, short, seqName, NEG_INF };
  }

  window.Tensors = { create };
})();
