// PRED-BiPN interface: method -> edge list -> run. Talks to the Flask routes
// /upload, /process, /task_status, /download and /visualize/compare.

// [backend name, display name, one-line description]
const METHODS = {
  proj: [
    ['Simple Weighted Projection', 'Simple Weighted Projection', 'Weight = number of shared neighbours'],
    ['One Mode Projection', 'One Mode Projection', 'An edge if any neighbour is shared'],
    ['Hypergraph Projection', 'Hypergraph Projection', 'Shared memberships as a hypergraph'],
    ['Jaccard Similarity Weighted Projection', 'Jaccard Similarity', 'Shared ÷ union of neighbours'],
    ['Cosine Similarity Weighted Projection', 'Cosine Similarity', 'Shared ÷ √(degree × degree)'],
    ['Adamic Adar Similarity Weighted Projection', 'Adamic–Adar Similarity', 'Shared neighbours weighted by 1 / log degree'],
    ['Resource Allocation Similarity Weighted Projection', 'Resource Allocation Similarity', 'Shared neighbours weighted by 1 / degree'],
    ['Odds Ratio Weighted Projection', 'Odds Ratio', '2×2 association test with p-values'],
    ['Pearson Correlation Weighted Projection', 'Pearson Correlation', 'Correlation of membership vectors'],
    ['Spearman Correlation Weighted Projection', 'Spearman Correlation', 'Rank correlation of membership vectors'],
    ['Kendall Correlation Weighted Projection', 'Kendall Correlation', 'Kendall τ of membership vectors'],
  ],
  bb: [
    ['Disparity Filter', 'Disparity Filter', 'Serrano et al. 2009 · edges significant for a node\'s strength'],
    ['Polya Filter', 'Pólya Filter', 'Marcaccioli & Livan 2019 · Pólya-urn null model'],
    ['Noise Corrected Filter', 'Noise Corrected Filter', 'Coscia & Neffke 2017 · binomial test of the weight'],
    ['Marginal Likelihood Filter', 'Marginal Likelihood Filter', 'Dianati 2016 · binomial test on integer weights'],
    ['Doubly Stochastic Filter', 'Doubly Stochastic Filter', 'Slater 2009 · strongest normalised edges until connected'],
    ['High Salience Skeleton', 'High Salience Skeleton', 'Grady et al. 2012 · edges on many shortest-path trees'],
    ['Locally Adaptive Network Sparsification Filter', 'Locally Adaptive Network Sparsification', 'Foti et al. 2011 · rank within each node'],
    ['Maximum Spanning Tree Filter', 'Maximum Spanning Tree', 'Kruskal · strongest tree over all nodes'],
    ['Edge Betweenness', 'Edge Betweenness', 'Brandes 2001 · betweenness ≥ factor × mean'],
    ['Modularity Filter', 'Modularity Filter', 'Rajeh et al. 2022 · keep nodes of largest modularity vitality'],
    ['Hierarchical Backbone', 'Hierarchical Backbone', 'BiCM null model · needs the bipartite file'],
    ['Global Threshold Filter', 'Global Threshold Filter', 'Keep edges above or below a fixed value'],
  ],
};

const ALPHA = { key: 'alpha', label: 'Significance level α', hint: 'Smaller keeps fewer edges', value: 0.05, min: 0.0001, max: 1, step: 0.005 };
const CORRECTION = { key: 'correction', label: 'Multiple-testing correction', type: 'select', value: 'fdr_bh',
  options: [['fdr_bh', 'Benjamini–Hochberg FDR'], ['bonferroni', 'Bonferroni'], ['none', 'None']] };
const DIRECTED = { key: 'directed', label: 'Network type', type: 'select', value: 'false',
  options: [['false', 'Undirected'], ['true', 'Directed']] };
const HEADER = { key: 'has_header', label: 'First row is a header', type: 'select', value: 'auto',
  options: [['auto', 'Auto'], ['true', 'Yes'], ['false', 'No']] };

const PARAMS = {
  'Disparity Filter': [ALPHA],
  'Doubly Stochastic Filter': [
    { key: 'ds_mode', label: 'Stopping rule', hint: 'Slater: add edges until connected', type: 'select', value: 'slater',
      options: [['slater', 'Slater (2009)'], ['strict', 'Strict (PRED-BiPN heuristic)']] },
    DIRECTED,
    { ...ALPHA, hint: 'Strict rule only', when: v => v.ds_mode === 'strict' }],
  'Locally Adaptive Network Sparsification Filter': [ALPHA],
  'Marginal Likelihood Filter': [ALPHA],
  'Noise Corrected Filter': [
    { key: 'delta', label: 'Threshold δ', hint: 'Keep score − δ·sd > 0', value: 3.09, min: 0, step: 0.01 },
    DIRECTED],
  'Modularity Filter': [
    { key: 'percentage', label: 'Share of nodes kept', hint: 'Largest |vitality|, 0 to 1', value: 0.3, min: 0.01, max: 1, step: 0.05 },
    { key: 'seed', label: 'Louvain seed', value: 42, min: 0, step: 1 },
    { key: 'keep_lcc', label: 'Keep only largest component', type: 'select', value: 'false', options: [['false', 'No'], ['true', 'Yes']] }],
  'Polya Filter': [ALPHA,
    { key: 'a_parameter', label: 'Parameter a', hint: '−1 = estimate from data, 1 = disparity filter', value: -1, min: -1, max: 15, step: 0.1 },
    { key: 'correction_method', label: 'Multiple-testing correction', type: 'select', value: 'fdr', options: [['fdr', 'Benjamini–Hochberg FDR'], ['bonferroni', 'Bonferroni']] },
    DIRECTED],
  'High Salience Skeleton': [{ key: 'threshold', label: 'Salience threshold', hint: '0 to 1', value: 0.9, min: 0, max: 1, step: 0.01 }],
  'Edge Betweenness': [
    { key: 'factor', label: 'Factor', hint: 'Keep betweenness ≥ factor × mean', value: 1, min: 0, step: 0.1 },
    { key: 'weighted', label: 'Use edge weights', hint: 'Length = 1 / weight', type: 'select', value: 'true', options: [['true', 'Yes'], ['false', 'No']] },
    { key: 'sample_size', label: 'Sampled source nodes', hint: '0 = exact', value: 0, min: 0, step: 1 }],
  'Hierarchical Backbone': [
    { key: 'z_threshold', label: 'Z-score threshold', value: 1.64, step: 0.01 },
    { key: 'alpha_percentile', label: 'Alpha percentile', hint: '1 to 100', value: 95, min: 1, max: 100, step: 1 }],
  'Global Threshold Filter': [
    { key: 'col_index', label: 'Column', type: 'select', value: '2', options: [] },
    { key: 'direction', label: 'Keep values', type: 'select', value: 'gte', options: [['gte', '≥ threshold'], ['lte', '≤ threshold']] },
    { key: 'threshold', label: 'Threshold', value: 0, step: 'any' }],
  'Cosine Similarity Weighted Projection': [{ key: 'threshold', label: 'Similarity threshold', hint: '0 to 1', value: 0.5, min: 0, max: 1, step: 0.01 }],
  'Adamic Adar Similarity Weighted Projection': [{ key: 'threshold', label: 'Similarity threshold', value: 0.5, min: 0, step: 0.01 }],
  'Resource Allocation Similarity Weighted Projection': [{ key: 'threshold', label: 'Similarity threshold', value: 0.5, min: 0, step: 0.01 }],
  'Jaccard Similarity Weighted Projection': [{ key: 'threshold', label: 'Jaccard threshold', hint: '0 to 1', value: 0, min: 0, max: 1, step: 0.01 }],
  'Kendall Correlation Weighted Projection': [CORRECTION, ALPHA],
  'Spearman Correlation Weighted Projection': [CORRECTION, ALPHA],
  'Pearson Correlation Weighted Projection': [CORRECTION, ALPHA],
  'Odds Ratio Weighted Projection': [CORRECTION, ALPHA, { key: 'or_threshold', label: 'Odds ratio threshold', value: 1, min: 0, step: 0.1 }],
};

const S = {
  cat: 'proj', method: null, picking: true, query: '',
  file: null,        // analysed input (see analyse)
  fileObj: null,     // File chosen by the user, uploaded on Run
  uploaded: null,    // server-side name of the input
  params: {},
  view: 'empty',     // empty | running | error | done
  run: null,
};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = n => n.toLocaleString('en-US');
const methodRow = name => [...METHODS.proj, ...METHODS.bb].find(m => m[0] === name);
const catOf = name => METHODS.proj.some(m => m[0] === name) ? 'proj' : 'bb';

/* ---------- reading an edge list in the browser ---------- */
const HEADER_WORDS = new Set(['source', 'target', 'from', 'to', 'node', 'node1', 'node2', 'gene', 'genes', 'gene1', 'gene2',
  'disease', 'diseases', 'src', 'dst', 'id', 'name', 'weight', 'score', 'value', 'col1', 'col2', 'edge', 'index']);
const isNum = x => x !== '' && x != null && !isNaN(Number(x));

// Same rules as backend/input_format.py, so "Detected" here matches what the run will do.
function analyse(text, name) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  const first = lines[0] || '';
  const sep = ['\t', ',', ';'].find(s => first.includes(s)) || null;
  const split = l => (sep ? l.split(sep) : l.trim().split(/\s+/)).map(c => c.trim());
  const p1 = split(first), p2 = lines[1] ? split(lines[1]) : [];
  const header = p1.length >= 2 && ((HEADER_WORDS.has(p1[0].toLowerCase()) && HEADER_WORDS.has(p1[1].toLowerCase()))
    || (p1.length > 2 && p2.length > 2 && !isNum(p1[2]) && isNum(p2[2])));
  const rows = lines.slice(header ? 1 : 0).map(split);
  const cols = Math.max(...rows.slice(0, 50).map(r => r.filter(c => c !== '').length), p1.length >= 2 ? 2 : p1.length);
  const A = new Set(), B = new Set();
  rows.forEach(r => { if (r[0]) A.add(r[0]); if (r[1]) B.add(r[1]); });
  let shared = 0; A.forEach(a => B.has(a) && shared++);
  const kind = cols < 2 ? 'invalid' : shared > 0.5 * Math.min(A.size, B.size) ? 'projection' : 'bipartite';
  const nearDup = [A, B].reduce((n, set) => {
    const seen = new Set(); let d = 0;
    set.forEach(x => { const k = x.replace(/[_\s]+$/, ''); seen.has(k) ? d++ : seen.add(k); });
    return n + d;
  }, 0);
  const info = { name, header, sep, cols, rows: rows.length, kind, A: A.size, B: B.size, nearDup,
    weighted: rows.slice(0, 50).every(r => isNum(r[2])), headerNames: header ? p1 : null,
    numericCols: p1.map((c, i) => i).filter(i => i >= 2 && rows.slice(0, 20).every(r => isNum(r[i]))),
    preview: rows.slice(0, 5), example: [rows[0]?.[0], rows[0]?.[1]] };
  if (kind === 'projection') Object.assign(info, oneMode(rows));
  return info;
}
function oneMode(rows) {
  const id = new Map(), parent = [];
  const get = x => { if (!id.has(x)) { id.set(x, parent.length); parent.push(parent.length); } return id.get(x); };
  const find = x => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
  const seen = new Set();
  rows.forEach(r => {
    if (!r[0] || !r[1] || r[0] === r[1]) return;
    const a = get(r[0]), b = get(r[1]);
    seen.add(a < b ? a + ',' + b : b + ',' + a);
    const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb;
  });
  let comps = 0; parent.forEach((p, i) => find(i) === i && comps++);
  return { nodes: parent.length, edges: seen.size, comps };
}

/* ---------- operation strip (schematic; the numbers are real) ---------- */
function rng(seed) { return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }
function denseNet(cut) {
  const r = rng(7), n = 16, p = [];
  for (let i = 0; i < n; i++) { const a = i / n * 6.283 + r() * .3, d = 30 + r() * 30; p.push([80 + Math.cos(a) * d, 60 + Math.sin(a) * d * .8]); }
  const e = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (r() < .32) e.push([i, j, r() < .2]);
  const kept = new Set(); e.forEach(([i, j, k]) => k && (kept.add(i), kept.add(j)));
  return `<svg viewBox="0 0 160 120" aria-hidden="true">${e.map(([i, j, k]) => `<line class="e${cut && !k ? ' cut' : ''}" x1="${p[i][0]}" y1="${p[i][1]}" x2="${p[j][0]}" y2="${p[j][1]}"/>`).join('')}${p.map((q, i) => `<circle class="${cut && !kept.has(i) ? 'nx' : 'nn'}" cx="${q[0]}" cy="${q[1]}" r="3.4"/>`).join('')}</svg>`;
}
function bipNet() {
  const r = rng(11), A = [...Array(6)].map((_, i) => [40, 14 + i * 18.4]), B = [...Array(5)].map((_, i) => [120, 22 + i * 19]);
  const e = []; A.forEach(a => B.forEach(b => r() < .42 && e.push([a, b])));
  return `<svg viewBox="0 0 160 120" aria-hidden="true">${e.map(([a, b]) => `<line class="e" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"/>`).join('')}${A.map(q => `<circle class="na" cx="${q[0]}" cy="${q[1]}" r="4"/>`).join('')}${B.map(q => `<circle class="nb" cx="${q[0]}" cy="${q[1]}" r="4"/>`).join('')}</svg>`;
}
function ringNet(cls, n, seed) {
  const r = rng(seed), p = [...Array(n)].map((_, i) => { const a = i / n * 6.283; return [80 + Math.cos(a) * 42, 60 + Math.sin(a) * 38]; });
  const e = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (r() < .45) e.push([p[i], p[j]]);
  return `<svg viewBox="0 0 160 120" aria-hidden="true">${e.map(([a, b]) => `<line class="e" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"/>`).join('')}${p.map(q => `<circle class="${cls}" cx="${q[0]}" cy="${q[1]}" r="4"/>`).join('')}</svg>`;
}
const ARROW = l => `<div class="arrow"><svg width="40" height="12" viewBox="0 0 40 12" aria-hidden="true"><path d="M0 6h36m-5-5 5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>${l}</div>`;
const panel = (svg, lbl, fig) => `<div class="panel">${svg}<div class="lbl">${esc(lbl)}</div><div class="fig">${fig || '—'}</div></div>`;
const ne = (n, e) => `${fmt(n)} <small>nodes</small><br>${fmt(e)} <small>edges</small>`;
const setNames = () => {
  const h = S.file?.headerNames;
  return h ? [h[0], h[1]].map(x => x.charAt(0).toUpperCase() + x.slice(1)) : ['Column 1 nodes', 'Column 2 nodes'];
};

function strip(kind, mode, figs = {}) {
  const [la, lb] = setNames();
  if (kind === 'proj') return `<div class="strip proj${mode ? '' : ' empty'}">
    ${panel(bipNet(), 'Bipartite input', figs.in)}${ARROW('project')}
    ${panel(ringNet('na', 7, 3), la, figs.a)}${panel(ringNet('nb', 5, 5), lb, figs.b)}</div>`;
  return `<div class="strip${mode ? '' : ' empty'}${mode === 'run' ? ' pruning' : ''}">
    ${panel(denseNet(false), 'Input network', figs.in)}${ARROW('filter')}${panel(denseNet(true), 'Backbone', figs.out)}</div>`;
}

/* ---------- left column ---------- */
function methodItems() {
  const q = S.query.toLowerCase();
  return METHODS[S.cat].filter(m => (m[1] + m[2]).toLowerCase().includes(q));
}
function listHTML() {
  return methodItems().map(m => `<button data-m="${esc(m[0])}"><b>${esc(m[1])}</b><span>${esc(m[2])}</span></button>`).join('')
    || '<p class="loading">No method matches.</p>';
}
function c1() {
  const m = S.method && methodRow(S.method);
  if (m && !S.picking) return `<div class="ch"><span class="step done">1</span><h2>Method</h2></div>
    <div class="cb"><div class="chosen"><div><b>${esc(m[1])}</b><span>${esc(m[2])}</span></div><button class="btn" data-act="change">Change</button></div></div>`;
  return `<div class="ch"><span class="step">1</span><h2>Method</h2><span class="aside" id="mcount">${methodItems().length} of ${METHODS[S.cat].length}</span></div>
    <div class="cb"><div class="seg" role="group" aria-label="Method type">
      <button data-cat="proj" aria-pressed="${S.cat === 'proj'}">Projections</button><button data-cat="bb" aria-pressed="${S.cat === 'bb'}">Backbone extraction</button></div>
      <input class="search" id="q" placeholder="Search methods" aria-label="Search methods" value="${esc(S.query)}">
      <div class="mlist" id="mlist">${listHTML()}</div></div>`;
}

function inputCheck() {
  const f = S.file, m = S.method;
  if (!f || !m) return null;
  if (f.kind === 'invalid') return { block: true, msg: `<b>This is not an edge list.</b> It has ${f.cols} column; an edge list needs at least two, source and target, with one edge per row.` };
  if (m === 'Hierarchical Backbone' && f.kind === 'projection')
    return { block: true, msg: '<b>Hierarchical Backbone needs the bipartite file.</b> This file is a projection: both columns hold the same kind of node. Upload the original two-column edge list, or choose another backbone method.', act: 'change' };
  if (catOf(m) === 'proj' && f.kind === 'projection')
    return { block: false, msg: '<b>This file looks like a projection already.</b> Both columns hold the same kind of node. A projection is made from the bipartite edge list.' };
  if (catOf(m) === 'bb' && m !== 'Hierarchical Backbone' && f.kind === 'bipartite')
    return { block: false, msg: '<b>Backbone filters work on a one-mode network.</b> This is a bipartite edge list. Run a projection first, then choose <i>Use for backbone</i> on its result.' };
  return null;
}
function c2() {
  const f = S.file;
  if (!f) return `<div class="ch"><span class="step">2</span><h2>Edge list</h2></div>
    <div class="cb"><div class="drop" id="drop">Drop a .txt, .tsv or .csv file here<br><button class="btn" data-act="pick">Choose file</button></div></div>`;
  const kind = { bipartite: 'Bipartite edge list', projection: 'One-mode network', invalid: 'Not an edge list' }[f.kind];
  const extra = f.kind === 'bipartite' ? `${fmt(f.A)} × ${fmt(f.B)} nodes` : f.kind === 'projection' ? `${fmt(f.nodes)} nodes${f.weighted ? ' · weighted' : ''}` : '';
  const chk = inputCheck();
  return `<div class="ch"><span class="step done">2</span><h2>Edge list</h2></div>
    <div class="cb"><div class="file"><div><b title="${esc(f.name)}">${esc(f.name)}</b><span>${fmt(f.rows)} rows · ${f.cols} column${f.cols === 1 ? '' : 's'} · ${f.header ? 'header row' : 'no header'}</span></div><button class="btn" data-act="pick"${S.view === 'running' ? ' disabled' : ''}>Replace</button></div>
    <div class="tags"><span class="tag ${f.kind === 'invalid' ? 'warn' : 'kind'}">${kind}</span>${extra ? `<span class="tag">${extra}</span>` : ''}${f.nearDup ? `<span class="tag warn" title="Names that differ only by a trailing _ or space count as different nodes">${fmt(f.nearDup)} near-duplicate names</span>` : ''}</div>
    ${chk ? `<div class="callout" role="alert"><p>${chk.msg}</p>${chk.act ? '<button class="btn" data-act="change">Choose another method</button>' : ''}</div>` : ''}</div>`;
}

function specs() {
  const list = [...(PARAMS[S.method] || []), HEADER];
  const vals = Object.fromEntries(list.map(s => [s.key, S.params[S.method + '|' + s.key] ?? s.value]));
  return list.filter(s => !s.when || s.when(vals)).map(s => {
    if (s.key === 'col_index') {
      const cols = S.file?.numericCols?.length ? S.file.numericCols : [2];
      const names = S.file?.headerNames;
      return { ...s, value: String(cols[0]), options: cols.map(i => [String(i), names?.[i] ? `${names[i]} (column ${i + 1})` : `Column ${i + 1}`]) };
    }
    if (s.key === 'has_header' && S.file) return { ...s, hint: `Detected: ${S.file.header ? 'yes' : 'no'}` };
    return s;
  });
}
const pval = s => S.params[S.method + '|' + s.key] ?? s.value;
function c3() {
  const chk = inputCheck();
  const ready = S.method && S.file && !(chk && chk.block) && S.view !== 'running';
  const fields = S.method ? specs().map(s => {
    const id = 'p_' + s.key, v = pval(s);
    const ctl = s.type === 'select'
      ? `<select id="${id}" data-p="${s.key}">${s.options.map(([o, t]) => `<option value="${esc(o)}"${String(v) === String(o) ? ' selected' : ''}>${esc(t)}</option>`).join('')}</select>`
      : `<input id="${id}" data-p="${s.key}" type="number" value="${esc(v)}"${s.min != null ? ` min="${s.min}"` : ''}${s.max != null ? ` max="${s.max}"` : ''} step="${s.step || 'any'}">`;
    return `<div class="field"><label for="${id}"><b>${esc(s.label)}</b>${s.hint ? `<span>${esc(s.hint)}</span>` : ''}</label>${ctl}</div>`;
  }).join('') : '';
  const name = S.method ? methodRow(S.method)[1] : 'method';
  return `<div class="ch"><span class="step">3</span><h2>Run</h2></div>
    <div class="cb">${fields}<div id="perr"></div>
    <button class="btn primary block" data-act="run" ${ready ? '' : 'disabled'}>${S.view === 'running' ? 'Running…' : 'Run ' + esc(name)}</button></div>`;
}

/* ---------- right column ---------- */
function logHTML(lines) {
  return lines.map(l => {
    const c = /^\[error\]/i.test(l) ? 'x' : /^\[warn/i.test(l) ? 'w' : '';
    return `<span${c ? ` class="${c}"` : ''}>${esc(l)}</span>`;
  }).join('\n');
}
const elapsed = () => { const s = Math.floor((Date.now() - S.run.t0) / 1000); return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };

function work() {
  const kind = S.method ? catOf(S.method) : S.cat, f = S.file, R = S.run;
  if (S.view === 'empty') {
    const figs = kind === 'proj' && f?.kind === 'bipartite' ? { in: `${fmt(f.rows)} <small>edges</small>`, a: `${fmt(f.A)} <small>nodes</small>`, b: `${fmt(f.B)} <small>nodes</small>` }
      : kind === 'bb' && f?.kind === 'projection' ? { in: ne(f.nodes, f.edges) } : {};
    const chk = inputCheck();
    const hint = !S.method ? 'Choose a method and an edge list. The result appears here.'
      : !f ? 'Add an edge list, then run.' : chk?.block ? 'Fix the input on the left to run.' : `Ready to run ${esc(methodRow(S.method)[1])}.`;
    return `${strip(kind, false, figs)}<p class="hint">${hint}</p>`;
  }
  const mname = esc(methodRow(R.method)[1]);
  if (S.view === 'running') return `<div class="status"><span class="dot run"></span><b>Running ${mname}</b><span class="aside num" id="clock">${elapsed()}</span></div>
    ${strip(catOf(R.method), 'run', catOf(R.method) === 'bb' && R.input.kind === 'projection' ? { in: ne(R.input.nodes, R.input.edges) } : {})}
    <div class="sect"><h3>Log</h3><div class="log" id="log">${logHTML(R.logs)}</div></div>`;
  if (S.view === 'error') return `<div class="err" role="alert"><b>${mname} stopped</b>${esc(R.error)}</div>
    <div class="sect"><details open><summary>Log</summary><div class="log">${logHTML(R.logs)}</div></details></div>`;

  // done
  const warns = R.logs.filter(l => /^\[warn/i.test(l)).length;
  const logBlock = `<div class="sect"><details><summary>Log${warns ? ` · ${warns} warning${warns > 1 ? 's' : ''}` : ''}</summary><div class="log">${logHTML(R.logs)}</div></details></div>`;
  const dl = (r, label) => `<a class="btn${label ? '' : ' primary'}" href="/download?file=${encodeURIComponent(r.name)}" download>${label || 'Download (.tsv)'}</a>`;
  const res = R.results;
  if (catOf(R.method) === 'proj' && res.length >= 2) {
    const [la, lb] = setNames(), [a, b] = res, inp = R.input;
    const figs = { in: inp.kind === 'bipartite' ? `${fmt(inp.rows)} <small>edges</small>` : '', a: a.stats ? ne(a.stats.nodes, a.stats.edges) : '', b: b.stats ? ne(b.stats.nodes, b.stats.edges) : '' };
    const row = (r, cls, label, ex) => `<div class="out ${cls}"><div><b>${esc(label)}</b><span class="num">${r.stats ? `${fmt(r.stats.nodes)} nodes · ${fmt(r.stats.edges)} edges` : esc(r.name)}</span>${ex ? `<span> · e.g. ${esc(ex)}</span>` : ''}</div>
      ${dl(r, 'Download (.tsv)')}<button class="btn" data-use="${esc(r.name)}">Use for backbone</button></div>`;
    return `<div class="status"><span class="dot"></span><b>${mname} finished</b><span class="aside">${res.length} output files</span></div>
      ${strip('proj', true, figs)}
      <div class="sect"><div class="outs">${row(a, 'a', la, inp.example[0])}${row(b, 'b', lb, inp.example[1])}${res.slice(2).map(r => row(r, '', 'Other output')).join('')}</div></div>${logBlock}`;
  }
  const r = res[0], inp = R.input, st = r.stats;
  const bb = catOf(R.method) === 'bb' && inp.kind === 'projection' && st && inp.edges > 0;
  const pct = bb && inp.edges ? st.edges / inp.edges * 100 : null;
  return `<div class="status"><span class="dot"></span><b>${mname} finished</b><span class="aside">${esc(r.name)}</span></div>
    ${bb ? strip('bb', true, { in: ne(inp.nodes, inp.edges), out: ne(st.nodes, st.edges) }) + `
    <div class="kept"><div class="track" role="img" aria-label="${pct.toFixed(1)} percent of edges kept"><i style="width:${Math.min(100, pct)}%"></i></div>
      <p><span><b class="num">${pct < 10 ? pct.toFixed(1) : pct.toFixed(0)} %</b> of edges kept</span><span>components <span class="num">${fmt(inp.comps)} → ${fmt(st.comps)}</span></span></p></div>`
      : st ? `<div class="sect"><p class="num">${fmt(st.nodes)} nodes · ${fmt(st.edges)} edges</p></div>` : ''}
    <div class="sect actions">${dl(r, catOf(R.method) === 'bb' ? 'Download backbone (.tsv)' : 'Download result (.tsv)').replace('class="btn"', 'class="btn primary"')}
      ${res.slice(1).map(x => dl(x, 'Download ' + esc(x.name.includes('communit') ? 'communities' : 'extra file'))).join('')}
      ${bb ? '<button class="btn" data-act="compare">Compare networks</button>' : ''}
      ${catOf(R.method) === 'proj' ? `<button class="btn" data-use="${esc(r.name)}">Use for backbone</button>` : ''}</div>
    <div id="cmp"></div>${logBlock}`;
}
function extra() {
  const f = S.file;
  if (!f || !f.preview.length) return '';
  const n = Math.min(3, f.cols);
  const head = f.headerNames ? f.headerNames.slice(0, n) : ['column 1', 'column 2', 'column 3'].slice(0, n);
  return `<div class="card" style="margin-top:12px"><div class="ch"><h2>File preview</h2><span class="aside">first ${f.preview.length} of ${fmt(f.rows)} rows</span></div>
    <div class="cb"><div class="tbl"><table><tr>${head.map((h, i) => `<th${i === 2 ? ' class="r"' : ''}>${esc(h)}</th>`).join('')}</tr>
    ${f.preview.map(r => `<tr>${r.slice(0, n).map((c, i) => `<td${i === 2 ? ' class="r"' : ''}>${esc(c ?? '')}</td>`).join('')}</tr>`).join('')}</table></div></div></div>`;
}

function render() {
  $('c1').innerHTML = c1(); $('c2').innerHTML = c2(); $('c3').innerHTML = c3();
  $('work').innerHTML = work(); $('extra').innerHTML = extra();
}

/* ---------- actions ---------- */
// A new input invalidates the server copy and any per-file choice (the Global
// Threshold column list depends on the file).
function setInput(f, fileObj, uploaded) {
  S.file = f; S.fileObj = fileObj; S.uploaded = uploaded; S.view = 'empty';
  for (const k in S.params) if (k.endsWith('|col_index')) delete S.params[k];
}
async function readFile(file) {
  if (S.view === 'running') return;   // the running job would otherwise store its upload as the new file's
  const f = analyse(await file.text(), file.name);
  setInput(f, file, null);
  if (!S.method) S.cat = f.kind === 'projection' ? 'bb' : 'proj';
  render();
}
function paramsForRun() {
  const out = {};
  for (const s of specs()) {
    const v = pval(s);
    if (s.type === 'select') out[s.key] = String(v);
    else if (v !== '' && !isNaN(parseFloat(v))) out[s.key] = parseFloat(v);
  }
  if (S.method === 'Global Threshold Filter') { out.col_index = parseInt(out.col_index, 10); out.chunk_size = 50000; }
  for (const s of specs()) {
    const v = out[s.key];
    if (s.type === 'select' || v == null) continue;
    if ((s.min != null && v < s.min) || (s.max != null && v > s.max))
      throw new Error(s.max != null ? `${s.label} must be between ${s.min} and ${s.max}.` : `${s.label} must be at least ${s.min}.`);
  }
  return out;
}
async function json(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.success === false) throw new Error(d.message || `Request failed (${r.status})`);
  return d;
}
async function run() {
  let params;
  try { params = paramsForRun(); } catch (e) { $('perr').innerHTML = `<p class="callout" role="alert">${esc(e.message)}</p>`; return; }
  S.run = { method: S.method, t0: Date.now(), logs: [], input: S.file, results: [] };
  S.view = 'running'; render();
  const clock = setInterval(() => { const c = $('clock'); if (c) c.textContent = elapsed(); }, 1000);
  try {
    if (!S.uploaded) {
      const fd = new FormData(); fd.append('file', S.fileObj); fd.append('method', S.method);
      S.uploaded = (await json('/upload', { method: 'POST', body: fd })).uploaded_file;
    }
    const { task_id } = await json('/process', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploaded_file: S.uploaded, method: S.method, ...params }) });
    let st;
    for (;;) {
      await new Promise(r => setTimeout(r, 800));
      st = await (await fetch(`/task_status/${task_id}`)).json();
      S.run.logs = st.logs || [];
      const log = $('log'); if (log) { log.innerHTML = logHTML(S.run.logs); log.scrollTop = log.scrollHeight; }
      if (st.status !== 'running') break;
    }
    if (st.status !== 'done') throw new Error(st.error || 'The run stopped without a result.');
    const names = Object.values(st.result?.files || {}).filter(v => typeof v === 'string' && v);
    if (!names.length) throw new Error('The run finished but returned no result file.');
    S.run.results = await Promise.all(names.map(async name => {
      try {
        const text = await (await fetch(`/download?file=${encodeURIComponent(name)}`)).text();
        const a = analyse(text, name);
        return { name, stats: a.kind === 'invalid' ? null : oneMode(text.split(/\r?\n/).filter(l => l.trim()).slice(a.header ? 1 : 0)
          .map(l => (a.sep ? l.split(a.sep) : l.trim().split(/\s+/)).map(c => c.trim()))) };
      } catch { return { name, stats: null }; }
    }));
    S.view = 'done';
  } catch (e) {
    S.run.error = e.message; S.view = 'error';
  } finally {
    clearInterval(clock); render();
  }
}
async function compare(btn) {
  const R = S.run, box = $('cmp');
  btn.disabled = true; box.innerHTML = '<div class="sect loading">Drawing both networks…</div>';
  try {
    const d = await json('/visualize/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ before_file: S.uploaded, after_file: R.results[0].name, method: R.method }) });
    const fig = (img, label, s) => `<figure><img src="data:image/png;base64,${img}" alt="${label}: ${s.nodes} nodes, ${s.edges} edges"><figcaption>${label} · ${fmt(s.nodes)} nodes · ${fmt(s.edges)} edges · density ${s.density.toFixed(4)}</figcaption></figure>`;
    box.innerHTML = `<div class="sect"><div class="compare">${fig(d.image_before, 'Input network', d.stats_before)}${fig(d.image_after, 'Backbone', d.stats_after)}</div></div>`;
  } catch (e) {
    box.innerHTML = `<div class="err" role="alert"><b>Comparison failed</b>${esc(e.message)}</div>`;
    btn.disabled = false;
  }
}
async function useForBackbone(name) {
  let text;
  try {
    const r = await fetch(`/download?file=${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(`the server answered ${r.status}`);
    text = await r.text();
  } catch (e) {
    S.run.error = `Could not load ${name}: ${e.message}. Download it and add it as the edge list instead.`;
    S.view = 'error'; render(); return;
  }
  setInput(analyse(text, name), null, name);   // already on the server: no new upload
  S.cat = 'bb'; S.method = null; S.picking = true; S.query = '';
  render(); scrollTo({ top: 0, behavior: 'smooth' });
}

document.addEventListener('click', e => {
  const t = e.target.closest('button'); if (!t || t.disabled) return;
  if (t.dataset.cat) { S.cat = t.dataset.cat; S.query = ''; render(); $('q')?.focus(); }
  else if (t.dataset.m) { S.method = t.dataset.m; S.picking = false; if (S.view !== 'running') S.view = 'empty'; render(); }
  else if (t.dataset.use) useForBackbone(t.dataset.use);
  else if (t.dataset.act === 'change') { S.picking = true; S.cat = S.method ? catOf(S.method) : S.cat; render(); $('q')?.focus(); }
  else if (t.dataset.act === 'pick') $('file').click();
  else if (t.dataset.act === 'run') run();
  else if (t.dataset.act === 'compare') compare(t);
});
document.addEventListener('input', e => {
  if (e.target.id === 'q') { S.query = e.target.value; $('mlist').innerHTML = listHTML(); $('mcount').textContent = `${methodItems().length} of ${METHODS[S.cat].length}`; }
  else if (e.target.dataset.p) S.params[S.method + '|' + e.target.dataset.p] = e.target.value;
});
document.addEventListener('change', e => {
  if (e.target.dataset.p) {
    S.params[S.method + '|' + e.target.dataset.p] = e.target.value;
    if (e.target.tagName === 'SELECT') { $('c3').innerHTML = c3(); $('p_' + e.target.dataset.p)?.focus(); }
  }
  if (e.target.id === 'file' && e.target.files[0]) { readFile(e.target.files[0]); e.target.value = ''; }
});
// A file dropped anywhere else would make the browser open it and lose the session.
document.addEventListener('dragover', e => { e.preventDefault(); e.target.closest?.('.drop')?.classList.add('over'); });
document.addEventListener('dragleave', e => e.target.closest?.('.drop')?.classList.remove('over'));
document.addEventListener('drop', e => {
  e.preventDefault();
  const d = e.target.closest?.('.drop'); if (!d) return;
  d.classList.remove('over');
  const f = e.dataTransfer.files[0]; if (f) readFile(f);
});
// dark is the default; the choice is remembered (see the inline script in login.html)
$('theme').onclick = () => {
  const r = document.documentElement, next = r.dataset.theme === 'light' ? 'dark' : 'light';
  if (next === 'dark') delete r.dataset.theme; else r.dataset.theme = 'light';
  try { localStorage.setItem('theme', next); } catch (e) {}
};
// Tool / Help / Info are sections of this page, switched by the URL hash
function showPage() {
  const page = ['help', 'info'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'tool';
  ['tool', 'help', 'info'].forEach(p => { $('page-' + p).hidden = p !== page; });
  document.querySelectorAll('nav a[data-page]').forEach(a => a.toggleAttribute('aria-current', a.dataset.page === page));
  if (page !== 'tool') scrollTo(0, 0);
}
addEventListener('hashchange', showPage);
showPage();
render();
