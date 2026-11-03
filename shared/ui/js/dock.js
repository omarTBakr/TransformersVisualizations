/*
 * Minimal dockable layout manager (no dependencies).
 *
 * Layout tree:
 *   split: { t: "row" | "col", s: [sizes...], c: [children...] }
 *   stack: { t: "stack", p: [panelIds...], a: activeIndex, m: collapsed? }
 *
 * Panels are existing DOM elements. They are moved (never re-created) into the layout,
 * so their state and event listeners survive any re-arrangement.
 *
 *   const dock = Dock.create(workspaceEl, {
 *     panels: { code: { el, title }, ... },
 *     presets: { default: tree, ... },
 *     storageKey: "arch-layout-v1",
 *     onResize: id => {},          // a panel changed size
 *     onChange: () => {},          // layout changed (for menus)
 *   });
 */
(function () {
  "use strict";

  const MIN = 0.06;        // smallest share a split child may shrink to
  const DRAG_START = 5;    // px before a tab press becomes a drag
  const clone = o => JSON.parse(JSON.stringify(o));

  function create(root, opts) {
    const panels = opts.panels;
    const ids = Object.keys(panels);
    const pool = document.createElement("div");
    pool.hidden = true;
    document.body.appendChild(pool);
    const mq = window.matchMedia("(max-width: 1000px)");
    let tree, closed = [], maximized = null;

    // ---------- persistence ----------
    function load() {
      try {
        const saved = JSON.parse(localStorage.getItem(opts.storageKey) || "null");
        if (saved && saved.tree) return { tree: saved.tree, closed: saved.closed || [] };
      } catch { /* ignore */ }
      return null;
    }
    function save() {
      try { localStorage.setItem(opts.storageKey, JSON.stringify({ tree, closed })); } catch { /* ignore */ }
    }

    // ---------- tree helpers ----------
    function walk(node, fn, parent = null, idx = -1) {
      fn(node, parent, idx);
      if (node.c) node.c.forEach((ch, i) => walk(ch, fn, node, i));
    }
    function stacks() { const out = []; walk(tree, n => n.t === "stack" && out.push(n)); return out; }
    function findStack(id) { return stacks().find(s => s.p.includes(id)); }
    function parentOf(target) { let res = null; walk(tree, (n, p, i) => { if (n === target) res = { p, i }; }); return res; }

    // Remove empty stacks, collapse single-child splits, merge same-direction nesting.
    function normalize(node) {
      if (node.t === "stack") return node.p.length ? node : null;
      const kids = [], sizes = [];
      node.c.forEach((ch, i) => {
        const n = normalize(ch);
        if (!n) return;
        if (n.t === node.t) {                         // merge e.g. row inside row
          const tot = n.s.reduce((a, b) => a + b, 0);
          n.c.forEach((g, j) => { kids.push(g); sizes.push(node.s[i] * n.s[j] / tot); });
        } else { kids.push(n); sizes.push(node.s[i]); }
      });
      if (!kids.length) return null;
      if (kids.length === 1) return kids[0];
      const tot = sizes.reduce((a, b) => a + b, 0);
      return { t: node.t, s: sizes.map(x => x / tot), c: kids };
    }

    // Make the tree contain exactly the known, non-closed panels.
    function sanitize(t) {
      const seen = new Set();
      walk(t, n => {
        if (n.t !== "stack") return;
        n.p = n.p.filter(id => panels[id] && !seen.has(id) && !closed.includes(id) && seen.add(id));
        n.a = Math.min(Math.max(0, n.a || 0), Math.max(0, n.p.length - 1));
      });
      let fixed = normalize(t) || { t: "stack", p: [], a: 0 };
      const missing = ids.filter(id => !seen.has(id) && !closed.includes(id));
      if (missing.length) {
        const first = (() => { let s = null; walk(fixed, n => { if (!s && n.t === "stack") s = n; }); return s; })();
        first.p.push(...missing);
      }
      return fixed;
    }

    function removePanel(id) {
      const st = findStack(id);
      if (!st) return;
      st.p.splice(st.p.indexOf(id), 1);
      st.a = Math.min(st.a, Math.max(0, st.p.length - 1));
    }

    function dropPanel(id, target, zone) {
      if (zone === "center") {
        removePanel(id);
        target.p.push(id);
        target.a = target.p.length - 1;
      } else {
        const dir = zone === "left" || zone === "right" ? "row" : "col";
        const before = zone === "left" || zone === "top";
        removePanel(id);
        const fresh = { t: "stack", p: [id], a: 0 };
        const loc = parentOf(target);
        if (loc && loc.p && loc.p.t === dir) {        // split the target's share within its parent
          const share = loc.p.s[loc.i] / 2;
          loc.p.s[loc.i] = share;
          loc.p.c.splice(before ? loc.i : loc.i + 1, 0, fresh);
          loc.p.s.splice(before ? loc.i : loc.i + 1, 0, share);
        } else {
          const copy = { t: target.t, p: target.p, a: target.a };
          const split = { t: dir, s: [0.5, 0.5], c: before ? [fresh, copy] : [copy, fresh] };
          if (!loc || !loc.p) tree = split;
          else loc.p.c[loc.i] = split;
        }
      }
      tree = sanitize(tree);
      maximized = null;
      render();
      save();
    }

    // ---------- rendering ----------
    // Moving elements in the DOM resets their scroll offsets, so remember and restore them.
    function saveScroll() {
      const out = [];
      ids.forEach(id => {
        const el = panels[id].el;
        [el, ...el.querySelectorAll("*")].forEach(n => {
          if (n.scrollTop || n.scrollLeft) out.push([n, n.scrollTop, n.scrollLeft]);
        });
      });
      return out;
    }

    function render() {
      const scroll = saveScroll();
      ids.forEach(id => pool.appendChild(panels[id].el));
      root.replaceChildren();
      root.classList.toggle("dk-mobile", mq.matches);
      if (mq.matches) {                               // small screens: simple vertical stack
        ids.filter(id => !closed.includes(id)).forEach(id => {
          const box = document.createElement("div");
          box.className = "dk-mobile-panel";
          box.innerHTML = `<div class="dk-tabs"><span class="dk-tab active">${panels[id].title}</span></div>`;
          const body = document.createElement("div");
          body.className = "dk-body";
          body.appendChild(panels[id].el);
          panels[id].el.hidden = false;
          box.appendChild(body);
          root.appendChild(box);
        });
      } else {
        root.appendChild(build(tree));
      }
      requestAnimationFrame(() => scroll.forEach(([n, t, l]) => { n.scrollTop = t; n.scrollLeft = l; }));
      if (opts.onChange) opts.onChange();
    }

    function build(node) {
      if (node.t === "stack") return buildStack(node);
      const el = document.createElement("div");
      el.className = `dk-split dk-${node.t}`;
      node.c.forEach((ch, i) => {
        if (i) el.appendChild(buildSash(node, i, el));
        const wrap = document.createElement("div");
        wrap.className = "dk-child";
        // A collapsed stack is a thin fixed strip; the others share the remaining space.
        wrap.style.flex = ch.t === "stack" && ch.m ? "0 0 32px" : `${node.s[i]} 1 0`;
        wrap.appendChild(build(ch));
        el.appendChild(wrap);
      });
      return el;
    }

    function buildCollapsed(st) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "dk-stack dk-min";
      el._stack = st;
      el.title = "Expand";
      el.innerHTML = `<span class="dk-min-icon">◂</span>${st.p.map(id => `<span class="dk-min-title">${panels[id].title}</span>`).join("")}<span class="dk-min-hint">click to open</span>`;
      el.addEventListener("click", () => {
        st.m = false;
        const loc = parentOf(st);                       // give it at least a quarter of its row/column
        if (loc && loc.p && loc.p.s[loc.i] < 0.25) {
          const tot = loc.p.s.reduce((a, b) => a + b, 0);
          loc.p.s = loc.p.s.map((x, k) => (k === loc.i ? 0.25 * tot : x * (0.75 * tot) / (tot - loc.p.s[loc.i])));
        }
        render(); save();
      });
      st.p.forEach(id => { panels[id].el.hidden = true; });
      return el;
    }

    function buildStack(st) {
      if (st.m) return buildCollapsed(st);
      const el = document.createElement("div");
      el.className = "dk-stack" + (maximized === st ? " dk-max" : "");
      el._stack = st;
      const head = document.createElement("div");
      head.className = "dk-tabs";
      st.p.forEach((id, i) => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "dk-tab" + (i === st.a ? " active" : "");
        tab.textContent = panels[id].title;
        tab.title = "Click to show · drag to move or dock";
        tab.addEventListener("pointerdown", e => startDrag(e, id, st, i));
        head.appendChild(tab);
      });
      const tools = document.createElement("span");
      tools.className = "dk-tools";
      tools.innerHTML = `<button type="button" class="dk-btn" data-act="min" title="Collapse to a thin strip">▁</button>
        <button type="button" class="dk-btn" data-act="max" title="${maximized === st ? "Restore" : "Maximize"}">${maximized === st ? "⤡" : "⤢"}</button>
        <button type="button" class="dk-btn" data-act="close" title="Close this panel (reopen from Layout)">×</button>`;
      tools.addEventListener("click", e => {
        const b = e.target.closest("[data-act]");
        if (!b) return;
        if (b.dataset.act === "max") { maximized = maximized === st ? null : st; render(); }
        else if (b.dataset.act === "min") { st.m = true; if (maximized === st) maximized = null; render(); save(); }
        else {
          const id = st.p[st.a];
          if (stacks().reduce((n, s) => n + s.p.length, 0) <= 1) return;  // keep at least one panel
          closed.push(id);
          removePanel(id);
          tree = sanitize(tree);
          render(); save();
        }
      });
      head.appendChild(tools);
      const body = document.createElement("div");
      body.className = "dk-body";
      st.p.forEach((id, i) => {
        panels[id].el.hidden = i !== st.a;
        body.appendChild(panels[id].el);
      });
      el.append(head, body);
      return el;
    }

    function buildSash(node, i, splitEl) {
      const sash = document.createElement("div");
      sash.className = "dk-sash";
      sash.addEventListener("pointerdown", e => {
        e.preventDefault();
        const horiz = node.t === "row";
        const rect = splitEl.getBoundingClientRect();
        const total = horiz ? rect.width : rect.height;
        const start = horiz ? e.clientX : e.clientY;
        const a0 = node.s[i - 1], b0 = node.s[i];
        const kids = splitEl.querySelectorAll(":scope > .dk-child");
        const move = ev => {
          const d = ((horiz ? ev.clientX : ev.clientY) - start) / total;
          const a = Math.min(Math.max(a0 + d, MIN), a0 + b0 - MIN);
          node.s[i - 1] = a; node.s[i] = a0 + b0 - a;
          kids[i - 1].style.flex = `${node.s[i - 1]} 1 0`;
          kids[i].style.flex = `${node.s[i]} 1 0`;
        };
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          window.removeEventListener("pointercancel", up);
          save();
          if (opts.onResize) ids.forEach(id => opts.onResize(id));
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
        window.addEventListener("pointercancel", up);
      });
      return sash;
    }

    // ---------- drag tabs to dock ----------
    const overlay = document.createElement("div");
    overlay.className = "dk-drop";
    const ghost = document.createElement("div");
    ghost.className = "dk-ghost";

    function startDrag(e, id, st, idx) {
      if (e.button !== 0) return;
      const x0 = e.clientX, y0 = e.clientY;
      let dragging = false, target = null, zone = null;
      e.preventDefault();
      const move = ev => {
        if (!dragging && Math.hypot(ev.clientX - x0, ev.clientY - y0) < DRAG_START) return;
        if (!dragging) {
          dragging = true;
          ghost.textContent = panels[id].title;
          document.body.append(overlay, ghost);
          document.body.classList.add("dk-dragging");
        }
        ghost.style.transform = `translate(${ev.clientX + 12}px, ${ev.clientY + 8}px)`;
        const under = document.elementFromPoint(ev.clientX, ev.clientY);
        const stEl = under && under.closest(".dk-stack:not(.dk-min)");
        if (!stEl) { overlay.style.display = "none"; target = null; return; }
        const r = stEl.getBoundingClientRect();
        const fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
        const edge = Math.min(fx, 1 - fx, fy, 1 - fy);
        zone = edge > 0.25 ? "center" : edge === fx ? "left" : edge === 1 - fx ? "right" : edge === fy ? "top" : "bottom";
        target = stEl._stack;
        if (target === st && st.p.length === 1) { overlay.style.display = "none"; target = null; return; }
        const z = { center: [0, 0, 1, 1], left: [0, 0, .5, 1], right: [.5, 0, .5, 1], top: [0, 0, 1, .5], bottom: [0, .5, 1, .5] }[zone];
        Object.assign(overlay.style, { display: "block", left: `${r.left + z[0] * r.width}px`, top: `${r.top + z[1] * r.height}px`,
                                       width: `${z[2] * r.width}px`, height: `${z[3] * r.height}px` });
      };
      const up = ev => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        window.removeEventListener("pointercancel", up);
        overlay.remove(); ghost.remove();
        document.body.classList.remove("dk-dragging");
        if (!dragging) { st.a = idx; render(); save(); return; }   // plain click: activate tab
        if (ev.type === "pointerup") move(ev);                         // final position decides the zone
        if (target) dropPanel(id, target, zone);
      };
      // Listen on window (not pointer capture) so the drop always fires wherever the pointer ends up.
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      window.addEventListener("pointercancel", up);
    }

    // ---------- resize notifications ----------
    if (window.ResizeObserver && opts.onResize) {
      const pending = new Set();
      let raf = 0;
      const ro = new ResizeObserver(entries => {
        entries.forEach(en => pending.add(en.target.dataset.panel));
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => { pending.forEach(id => opts.onResize(id)); pending.clear(); });
      });
      ids.forEach(id => ro.observe(panels[id].el));
    }
    mq.addEventListener("change", render);

    // ---------- public API ----------
    const api = {
      preset(name) {
        closed = [];
        tree = sanitize(clone(opts.presets[name]));
        maximized = null;
        render(); save();
      },
      reset() { api.preset("default"); },
      closedPanels: () => closed.slice(),
      open(id) {
        closed = closed.filter(x => x !== id);
        const st = stacks().sort((a, b) => b.p.length - a.p.length)[0];
        st.p.push(id); st.a = st.p.length - 1;
        render(); save();
      },
      show(id) {                                     // bring a panel's tab to the front
        const st = findStack(id);
        if (!st || st.p[st.a] === id) return;
        st.a = st.p.indexOf(id);
        render(); save();
      },
      isVisible(id) {
        const st = findStack(id);
        return !!st && st.p[st.a] === id;
      },
    };

    const saved = load();
    closed = saved ? saved.closed.filter(id => panels[id]) : [];
    tree = sanitize(saved ? saved.tree : clone(opts.presets.default));
    render();
    return api;
  }

  window.Dock = { create };
})();
