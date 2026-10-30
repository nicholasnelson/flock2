/* FLOCK 2.0 replay viewer: a top-down, zoomable, pannable map of a recorded meadow run.
   Reads the .flock.gz files scripts/m2_record.py writes (see README.md for the format) and index.json
   for the run menu. Vanilla JS and canvas, no build step. World coordinates are tiles: x = column,
   y = row (north, the sim's -row, is up); the sim's heading h points along (row, col) = (-cos h, sin h),
   so on screen it is the angle h measured clockwise from up. */
'use strict';

const params = new URLSearchParams(location.search);
const REPLAY_DIR = params.get('dir') || '../../docs/flock-2/meadow2-results/replays/';
let N = 48;                   // tiles per side (set from each replay's header; the river meadow is 72)
const TAU = Math.PI * 2;
const FOV = 300 * Math.PI / 180;
const SPEED_STEPS_PER_S = 60; // 1x: one sim step per frame at 60 Hz
const PX_PER_TILE_TERRAIN = 8;

const $ = (id) => document.getElementById(id);
const map = $('map'), mctx = map.getContext('2d');
const scrub = $('scrub'), sctx = scrub.getContext('2d');
const tooltip = $('tooltip'), status = $('status');

// ---------------------------------------------------------------- state
let R = null;        // the loaded replay: {header, arrays, T, S, meta, ...}
const view = { s: 12, ox: 0, oy: 0, fitted: false };    // px per tile, pan offset in px; fitted once the canvas has a size
const play = { step: 0, playing: false, speed: 1, acc: 0, last: 0 };
const opt = {
  fill: 'action', trail: 200, fov: false, rings: true, labels: true, ring: true, mud: true, tint: true, follow: false,
};
let selected = -1, hover = null, dpr = window.devicePixelRatio || 1;
let grassCanvas = null, grassIdx = -1, terrainCanvas = null, waterDistField = null;
let pal = null, eventsSorted = [];

// ---------------------------------------------------------------- palette from CSS
function palette() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  pal = {
    grass0: v('--grass0'), grass1: v('--grass1'), water: v('--water'), waterEdge: v('--water-edge'), mud: v('--mud'), fence: v('--fence'),
    night: v('--night'), fog: v('--fog'), tree: v('--tree'), willow: v('--willow'), hut: v('--hut'), salt: v('--salt'), flower: v('--flower'),
    mushroom: v('--mushroom'), sheep: v('--sheep'), sheepRest: v('--sheep-rest'), sheepDead: v('--sheep-dead'), sheepEdge: v('--sheep-edge'),
    trail: v('--trail'), ringOk: v('--ring-ok'), ringLow: v('--ring-low'), ringBad: v('--ring-bad'), death: v('--death'), drink: v('--drink'),
    near: v('--near'), select: v('--select'), dayBand: v('--day-band'), nightBand: v('--night-band'), fogBand: v('--fog-band'),
    text: v('--text'), dim: v('--dim'), line: v('--line'), panel: v('--panel'), accent: v('--accent'), accent2: v('--accent2'),
  };
  pal.grassRGB = [hexToRGB(pal.grass0), hexToRGB(pal.grass1)];
  pal.waterRGB = hexToRGB(pal.water);
  pal.mudRGB = hexToRGB(pal.mud);
  grassIdx = -1; terrainCanvas = null;   // re-render the baked layers in the new colours
  buildLegend();
}
function hexToRGB(h) {
  h = h.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const ACTION_COLOURS = { stop: '#9aa0a8', walk: '#f2c14e', trot: '#f28c28', left: '#d9d9d9', right: '#d9d9d9', graze: '#5cc45c', drink: '#4aa3ff', rest: '#8f7bd6' };
const ID_COLOURS = Array.from({ length: 12 }, (_, i) => `hsl(${(i * 137.5) % 360} 70% 55%)`);
function levelColour(v) {  // 0..1.15 -> red / orange / blue
  if (v < 0.25) return pal.ringBad;
  if (v < 0.5) return pal.ringLow;
  return pal.ringOk;
}

// ---------------------------------------------------------------- loading
const DTYPES = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array, float64: Float64Array };

async function fetchReplay(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText} for ${url}`);
  let buf = await resp.arrayBuffer();
  const u8 = new Uint8Array(buf);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {   // gzip: the server sent the file as stored
    if (typeof DecompressionStream === 'undefined') throw new Error('this browser has no DecompressionStream; use a current Chrome, Edge, Firefox or Safari');
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return parseReplay(buf);
}

function parseReplay(buf) {
  const dv = new DataView(buf);
  if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'FLK2') throw new Error('not a flock2 replay');
  const hlen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
  const base = 8 + hlen + ((8 - ((8 + hlen) % 8)) % 8);
  const arrays = {};
  for (const [name, a] of Object.entries(header.arrays)) {
    const T = DTYPES[a.dtype];
    if (!T) throw new Error(`unknown dtype ${a.dtype}`);
    arrays[name] = new T(buf, base + a.offset, a.shape.reduce((p, q) => p * q, 1));
  }
  const T = header.arrays.pos.shape[0], S = header.arrays.pos.shape[1];
  const r = { header, arrays, T, S, meta: header.meta, world: header.world, sc: header.scales, grassSteps: header.grass_steps, events: header.events };
  r.DAY = header.meta.DAY || 2400;
  // decoded accessors
  r.posX = (t, s) => arrays.pos[(t * S + s) * 2 + 1] / r.sc.pos;   // column
  r.posY = (t, s) => arrays.pos[(t * S + s) * 2] / r.sc.pos;       // row
  r.heading = (t, s) => arrays.heading[t * S + s] / r.sc.heading * TAU;
  r.alive = (t, s) => (arrays.flags[t * S + s] & 1) !== 0;
  r.resting = (t, s) => (arrays.flags[t * S + s] & 2) !== 0;
  r.drank = (t, s) => (arrays.flags[t * S + s] & 4) !== 0;
  r.grazed = (t, s) => (arrays.flags[t * S + s] & 8) !== 0;
  r.action = (t, s) => arrays.action[t * S + s];
  r.level = (t, s, i) => arrays.levels[(t * S + s) * 4 + i] / r.sc.levels;
  r.fat = (t, s) => arrays.fat[t * S + s] / r.sc.fat;
  r.wdist = (t, s) => arrays.wdist[t * S + s] / r.sc.wdist;
  r.t = (t) => arrays.t[t];
  r.light = (t) => arrays.light[t] / r.sc.light;
  r.night = (t) => (arrays.step_flags[t] & 1) !== 0;
  r.fog = (t) => (arrays.step_flags[t] & 2) !== 0;
  r.HYD = header.meta.drive_names.indexOf('hydration');
  r.ENE = header.meta.drive_names.indexOf('energy');
  r.actionNames = header.meta.action_names;
  return r;
}

async function loadIndex() {
  const sel = $('run');
  try {
    const resp = await fetch(REPLAY_DIR + 'index.json', { cache: 'no-cache' });
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    const idx = await resp.json();
    sel.innerHTML = '';
    for (const e of idx.replays) {
      const o = document.createElement('option');
      o.value = e.file;
      o.textContent = e.label + (e.meta && e.meta.deaths !== undefined ? `  (${(e.meta.days || 0).toFixed(0)} d, ${e.meta.deaths} dead)` : '');
      o.title = e.description || '';
      sel.appendChild(o);
    }
    if (!idx.replays.length) { sel.innerHTML = '<option value="">no replays in index.json</option>'; return; }
    const want = (location.hash.match(/run=([^&]+)/) || [])[1];
    const pick = idx.replays.find((e) => e.file === decodeURIComponent(want || '')) || idx.replays[0];
    sel.value = pick.file;
    await loadRun(pick.file);
  } catch (err) {
    sel.innerHTML = '<option value="">index.json not found</option>';
    showStatus(`Could not load ${REPLAY_DIR}index.json: ${err.message}.\nServe the repository root with "python -m http.server 8765" and open http://localhost:8765/web/replay/`, true);
  }
}

async function loadRun(file) {
  play.playing = false; updatePlayButton();
  showStatus(`loading ${file} ...`);
  try {
    R = await fetchReplay(REPLAY_DIR + file);
  } catch (err) {
    showStatus(`Could not load ${file}: ${err.message}`, true);
    return;
  }
  hideStatus();
  N = R.world.n_tiles || 48;
  selected = -1; hover = null; grassIdx = -1; terrainCanvas = null; waterDistField = null;
  play.step = 0;
  const m = (location.hash.match(/step=(\d+)/) || [])[1];
  if (m && location.hash.includes(encodeURIComponent(file))) play.step = Math.min(R.T - 1, +m);
  $('desc').textContent = R.meta.description || '';
  $('meta').textContent = metaText();
  const trail = $('trail');
  trail.max = Math.min(2400, R.T);
  eventsSorted = allEvents();
  buildEventList();
  buildSheepTable();
  resetView();
  buildTerrain();
  resizeScrub();
  setHash();
  requestAnimationFrame(frame);
}

function metaText() {
  const m = R.meta, w = R.world;
  const rows = [
    ['label', m.label], ['policy', m.policy + (m.agent ? '  ' + m.agent : '')], ['days', `${m.days} (${m.steps} steps, DAY = ${m.DAY})`],
    ['sheep', m.sheep], ['willows / pond', m.willows], ['mud inner / edge', `${m.inner} / ${m.edge}`], ['fog probability', m.fog],
    ['fog days', w.fog_days.map((f, i) => (f ? 'D' + i : null)).filter(Boolean).join(' ') || 'none'],
    ['thirst scale', m.thirst], ['trample', m.trample], ['zero column', m.zero_column], ['meadow seed', `${m.meadow_seed} (index ${m.meadow_index})`],
    ['policy seed', m.seed], ['grass snapshot every', m.grass_every + ' steps'], ['deaths', `${m.deaths} (alive at end ${m.alive_at_end})`],
    ['ponds', w.ponds.map((p) => `(${p.row}, ${p.col}) r ${p.radius}`).join('; ')],
    ['git', m.git], ['recorded', m.date],
  ];
  return rows.map(([k, v]) => `${k.padEnd(22)} ${v}`).join('\n');
}

function showStatus(msg, isError) { status.textContent = msg; status.hidden = false; status.style.color = isError ? 'var(--death)' : ''; }
function hideStatus() { status.hidden = true; }

// ---------------------------------------------------------------- events
function allEvents() {
  const ev = [];
  for (const d of R.events.deaths) ev.push({ step: d.step, kind: 'death', sheep: d.sheep, text: `sheep ${d.sheep} died of ${d.cause} at (${d.pos[0]}, ${d.pos[1]})` });
  for (const n of R.events.first_near_water) ev.push({ step: n.step, kind: 'near', sheep: n.sheep, text: `sheep ${n.sheep} first within 8 tiles of water` });
  for (const b of R.events.drinks) ev.push({ step: b.step, end: b.end, kind: 'drink', sheep: b.sheep, text: `sheep ${b.sheep} drank ${b.sips} sips` });
  ev.sort((a, b) => a.step - b.step);
  return ev;
}
function buildEventList() {
  const box = $('events');
  box.innerHTML = '';
  const major = eventsSorted.filter((e) => e.kind !== 'drink');
  const head = document.createElement('div');
  head.className = 'dim';
  head.textContent = `${R.events.deaths.length} deaths, ${R.events.drinks.length} drinking bouts, ${R.events.first_near_water.length} first approaches. Click to jump.`;
  head.style.cursor = 'default';
  box.appendChild(head);
  for (const e of major) {
    const d = document.createElement('div');
    d.textContent = `${stepLabel(e.step)}  ${e.text}`;
    d.onclick = () => { setStep(e.step); selected = e.sheep; };
    box.appendChild(d);
  }
}
function stepLabel(step) {
  const t = R.t(step);
  return `D${Math.floor(t / R.DAY)} ${clockText(t)} (step ${step})`;
}
function clockText(t) {
  // dawn (step 0 of the day) is called 06:00; the sim's day is 60 % light, dusk at 0.6 DAY
  const frac = (t % R.DAY) / R.DAY;
  const h = (6 + 24 * frac) % 24;
  const hh = Math.floor(h), mm = Math.floor((h - hh) * 60);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}
function jumpEvent(dir) {
  const major = eventsSorted.filter((e) => e.kind !== 'drink');
  if (!major.length) return;
  let e = dir > 0 ? major.find((x) => x.step > play.step) : [...major].reverse().find((x) => x.step < play.step);
  if (!e) e = dir > 0 ? major[0] : major[major.length - 1];
  setStep(e.step); selected = e.sheep;
}

// ---------------------------------------------------------------- view
function resizeMap() {
  dpr = window.devicePixelRatio || 1;
  const w = map.clientWidth, h = map.clientHeight;
  if (map.width !== Math.round(w * dpr) || map.height !== Math.round(h * dpr)) {
    map.width = Math.round(w * dpr); map.height = Math.round(h * dpr);
  }
}
function resetView() {
  resizeMap();
  const w = map.clientWidth, h = map.clientHeight;
  if (w < 10 || h < 10) { view.fitted = false; return; }   // no layout yet: fit on the first drawn frame
  view.s = Math.max(2, Math.min(w, h) / (N + 2));
  view.ox = (w - N * view.s) / 2;
  view.oy = (h - N * view.s) / 2;
  view.fitted = true;
}
function zoomAt(sx, sy, f) {
  const ns = Math.min(400, Math.max(2, view.s * f));
  f = ns / view.s;
  view.ox = sx - (sx - view.ox) * f;
  view.oy = sy - (sy - view.oy) * f;
  view.s = ns;
}
function toWorld(sx, sy) { return [(sx - view.ox) / view.s, (sy - view.oy) / view.s]; }

// ---------------------------------------------------------------- baked layers
function buildTerrain() {
  // water (the sim's bilinear water_at cut at one half, so shorelines are curves) and the mud ring
  const px = N * PX_PER_TILE_TERRAIN;
  const c = document.createElement('canvas');
  c.width = px; c.height = px;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(px, px);
  const water = R.arrays.water, rm = R.arrays.regrow_map, nc = R.world.n_cells || 96, res = R.world.grass_res || 2;
  const [wr, wg, wb] = pal.waterRGB, [mr, mg, mb] = pal.mudRGB;
  const edge = hexToRGB(pal.waterEdge);
  for (let j = 0; j < px; j++) {
    const row = (j + 0.5) / PX_PER_TILE_TERRAIN;
    for (let i = 0; i < px; i++) {
      const col = (i + 0.5) / PX_PER_TILE_TERRAIN;
      const v = bilinear(water, N, row, col, 1);
      const k = (j * px + i) * 4;
      if (v >= 0.5) {
        const deep = Math.min(1, (v - 0.5) / 0.3);
        img.data[k] = edge[0] + (wr - edge[0]) * deep; img.data[k + 1] = edge[1] + (wg - edge[1]) * deep; img.data[k + 2] = edge[2] + (wb - edge[2]) * deep;
        img.data[k + 3] = 255;
      } else if (opt.mud) {
        const ci = Math.min(nc - 1, Math.floor(row * res)), cj = Math.min(nc - 1, Math.floor(col * res));
        const m = 1 - rm[ci * nc + cj] / R.sc.regrow_map;
        if (m > 0.01) {
          img.data[k] = mr; img.data[k + 1] = mg; img.data[k + 2] = mb; img.data[k + 3] = Math.round(150 * m);
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  terrainCanvas = c;
}
function bilinear(field, n, row, col, res) {
  // the sim's _bilinear: grid points at (i + 0.5) / res
  const qr = row * res - 0.5, qc = col * res - 0.5;
  const fr = Math.floor(qr), fc = Math.floor(qc), ur = qr - fr, uc = qc - fc;
  const g = (r, c) => field[Math.min(n - 1, Math.max(0, r)) * n + Math.min(n - 1, Math.max(0, c))];
  return g(fr, fc) * (1 - ur) * (1 - uc) + g(fr, fc + 1) * (1 - ur) * uc + g(fr + 1, fc) * ur * (1 - uc) + g(fr + 1, fc + 1) * ur * uc;
}
function grassSnapshotIndex(step) {
  const gs = R.grassSteps;
  let k = 0;
  while (k + 1 < gs.length && gs[k + 1] <= step) k++;
  return k;
}
function buildGrass(k) {
  const nc = R.world.n_cells || 96;
  if (!grassCanvas) { grassCanvas = document.createElement('canvas'); grassCanvas.width = nc; grassCanvas.height = nc; }
  const ctx = grassCanvas.getContext('2d');
  const img = ctx.createImageData(nc, nc);
  const g = R.arrays.grass, off = k * nc * nc, scale = R.sc.grass * 3;
  const [a, b] = pal.grassRGB;
  for (let i = 0; i < nc * nc; i++) {
    const u = Math.min(1, g[off + i] / scale);
    const w = Math.sqrt(u);  // the eye needs the low end stretched
    img.data[i * 4] = a[0] + (b[0] - a[0]) * w; img.data[i * 4 + 1] = a[1] + (b[1] - a[1]) * w; img.data[i * 4 + 2] = a[2] + (b[2] - a[2]) * w; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  grassIdx = k;
}

// ---------------------------------------------------------------- drawing
function frame(ts) {
  if (!R) return;
  if (play.playing) {
    const dt = Math.min(0.1, (ts - play.last) / 1000 || 0);
    play.acc += play.speed * SPEED_STEPS_PER_S * dt;
    const n = Math.floor(play.acc);
    if (n > 0) { play.acc -= n; play.step += n; }
    if (play.step >= R.T - 1) { play.step = R.T - 1; play.playing = false; updatePlayButton(); setHash(); }
  }
  play.last = ts;
  draw();
  requestAnimationFrame(frame);
}

function draw() {
  resizeMap();
  const step = play.step, ctx = mctx, w = map.clientWidth, h = map.clientHeight;
  if (!view.fitted) resetView();
  if (!view.fitted) return;
  if (!terrainCanvas) buildTerrain();
  const k = grassSnapshotIndex(step);
  if (k !== grassIdx) buildGrass(k);
  if (opt.follow && selected >= 0) {
    view.ox = w / 2 - R.posX(step, selected) * view.s;
    view.oy = h / 2 - R.posY(step, selected) * view.s;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.translate(view.ox, view.oy);
  ctx.scale(view.s, view.s);
  const s = view.s, lw = (px) => px / s;   // a line width of px pixels, in tiles

  // ground
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(grassCanvas, 0, 0, N, N);
  ctx.drawImage(terrainCanvas, 0, 0, N, N);
  // night and fog
  if (opt.tint) {
    const light = R.light(step);
    if (light < 0.98) { ctx.globalAlpha = (1 - light) * 0.55; ctx.fillStyle = pal.night; ctx.fillRect(0, 0, N, N); ctx.globalAlpha = 1; }
    if (R.fog(step)) { ctx.globalAlpha = 0.28; ctx.fillStyle = pal.fog; ctx.fillRect(0, 0, N, N); ctx.globalAlpha = 1; }
  }
  // the fence: the meadow's edge, and the half-tile band no sheep may enter
  ctx.strokeStyle = pal.fence; ctx.lineWidth = lw(2.5); ctx.strokeRect(0, 0, N, N);
  ctx.setLineDash([lw(4), lw(4)]); ctx.lineWidth = lw(1); ctx.globalAlpha = 0.5; ctx.strokeRect(0.5, 0.5, N - 1, N - 1); ctx.setLineDash([]); ctx.globalAlpha = 1;
  // distance rings round each pond: the bins the memory rule reads
  if (opt.rings) {
    ctx.lineWidth = lw(1); ctx.setLineDash([lw(6), lw(5)]); ctx.strokeStyle = pal.text; ctx.globalAlpha = 0.45;
    for (const p of R.world.ponds) {
      for (const d of [8, 12, 16]) {
        ctx.beginPath(); ctx.arc(p.col, p.row, p.radius + d, 0, TAU); ctx.stroke();
      }
    }
    const rv = R.world.river;
    if (rv) {
      // the river meadow (spec section 21.8): the bins are bands parallel to the river, on both sides
      const ax = -rv.nrow, ay = rv.ncol;   // along the river, in (x = col, y = row)
      for (const sign of [1, -1]) for (const d of [8, 12, 16]) {
        const off = sign * (rv.width / 2 + d), cx = rv.col + off * rv.ncol, cy = rv.row + off * rv.nrow;
        ctx.beginPath(); ctx.moveTo(cx - ax * 2 * N, cy - ay * 2 * N); ctx.lineTo(cx + ax * 2 * N, cy + ay * 2 * N); ctx.stroke();
      }
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    if (s > 7) {
      ctx.fillStyle = pal.text; ctx.globalAlpha = 0.6; ctx.font = `${lw(11)}px system-ui, sans-serif`; ctx.textAlign = 'center';
      for (const p of R.world.ponds) for (const d of [8, 12, 16]) ctx.fillText(String(d), p.col, p.row - p.radius - d - lw(3));
      if (rv) for (const sign of [1, -1]) for (const d of [8, 12, 16]) {
        const off = sign * (rv.width / 2 + d);
        ctx.fillText(String(d), rv.col + off * rv.ncol, rv.row + off * rv.nrow);
      }
      ctx.globalAlpha = 1;
    }
  }
  drawObjects(ctx, lw);
  drawTrails(ctx, step, lw);
  if (opt.fov) for (let i = 0; i < R.S; i++) if (R.alive(step, i) && i !== selected) drawFov(ctx, step, i, 0.06, lw, false);
  if (selected >= 0 && R.alive(step, selected)) drawFov(ctx, step, selected, opt.fov ? 0.12 : 0.08, lw, true);
  for (let i = 0; i < R.S; i++) if (i !== selected) drawSheep(ctx, step, i, lw);
  if (selected >= 0) drawSheep(ctx, step, selected, lw);
  // the hovered thing
  if (hover && hover.kind === 'sheep') {
    ctx.strokeStyle = pal.text; ctx.lineWidth = lw(1.5); ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.arc(R.posX(step, hover.i), R.posY(step, hover.i), 0.8, 0, TAU); ctx.stroke(); ctx.globalAlpha = 1;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  updateClock(step);
  updateSheepTable(step);
  drawScrub();
}

function objectName(kind) { return R.world.objects.kind_names[String(kind)] || `kind ${kind}`; }
function drawObjects(ctx, lw) {
  const o = R.world.objects;
  for (let i = 0; i < o.kind.length; i++) {
    if (!o.present[i]) continue;
    const name = objectName(o.kind[i]);
    const [row, col] = o.pos[i];
    ctx.lineWidth = lw(1); ctx.strokeStyle = pal.sheepEdge;
    switch (name) {
      case 'tree':
        ctx.fillStyle = pal.tree; ctx.beginPath(); ctx.arc(col, row, 0.55, 0, TAU); ctx.fill(); ctx.stroke(); break;
      case 'willow':
        ctx.fillStyle = pal.willow; ctx.beginPath(); ctx.arc(col, row, 0.6, 0, TAU); ctx.fill(); ctx.stroke();
        ctx.strokeStyle = pal.tree; ctx.lineWidth = lw(1.2);
        for (let a = 0; a < 6; a++) { const th = a * TAU / 6; ctx.beginPath(); ctx.moveTo(col, row); ctx.lineTo(col + 0.5 * Math.cos(th), row + 0.5 * Math.sin(th)); ctx.stroke(); }
        break;
      case 'hut':
        ctx.fillStyle = pal.hut; ctx.fillRect(col - 0.7, row - 0.7, 1.4, 1.4); ctx.strokeRect(col - 0.7, row - 0.7, 1.4, 1.4);
        ctx.beginPath(); ctx.moveTo(col - 0.7, row - 0.1); ctx.lineTo(col, row - 0.75); ctx.lineTo(col + 0.7, row - 0.1); ctx.stroke(); break;
      case 'salt':
        ctx.fillStyle = pal.salt; ctx.beginPath(); ctx.moveTo(col, row - 0.45); ctx.lineTo(col + 0.45, row); ctx.lineTo(col, row + 0.45); ctx.lineTo(col - 0.45, row); ctx.closePath(); ctx.fill(); ctx.stroke(); break;
      case 'flower':
        ctx.fillStyle = pal.flower;
        for (let a = 0; a < 5; a++) { const th = a * TAU / 5; ctx.beginPath(); ctx.arc(col + 0.18 * Math.cos(th), row + 0.18 * Math.sin(th), 0.13, 0, TAU); ctx.fill(); }
        ctx.fillStyle = '#ffe66d'; ctx.beginPath(); ctx.arc(col, row, 0.1, 0, TAU); ctx.fill(); break;
      case 'mushroom':
        ctx.fillStyle = pal.mushroom; ctx.beginPath(); ctx.arc(col, row, 0.3, Math.PI, TAU); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.fillRect(col - 0.08, row, 0.16, 0.25); break;
      default:
        ctx.fillStyle = pal.dim; ctx.beginPath(); ctx.arc(col, row, 0.3, 0, TAU); ctx.fill(); ctx.stroke();
    }
  }
}

function sheepColour(step, i) {
  if (!R.alive(step, i)) return pal.sheepDead;
  switch (opt.fill) {
    case 'id': return ID_COLOURS[i % ID_COLOURS.length];
    case 'action': return ACTION_COLOURS[R.actionNames[R.action(step, i)]] || pal.sheep;
    case 'hydration': return levelColour(R.level(step, i, R.HYD));
    case 'energy': return levelColour(R.level(step, i, R.ENE));
    case 'fat': return levelColour(R.fat(step, i));
    default: return R.resting(step, i) ? pal.sheepRest : pal.sheep;
  }
}

function drawSheep(ctx, step, i, lw) {
  const x = R.posX(step, i), y = R.posY(step, i), h = R.heading(step, i);
  const dx = Math.sin(h), dy = -Math.cos(h);         // the heading on screen
  const px = -dy, py = dx;                            // its perpendicular
  const alive = R.alive(step, i), resting = R.resting(step, i), isSel = i === selected;
  if (isSel) {
    ctx.strokeStyle = pal.select; ctx.lineWidth = lw(3); ctx.beginPath(); ctx.arc(x, y, 0.75, 0, TAU); ctx.stroke();
  }
  ctx.fillStyle = sheepColour(step, i);
  ctx.strokeStyle = pal.sheepEdge; ctx.lineWidth = lw(1);
  if (alive) {
    ctx.beginPath();
    ctx.moveTo(x + 0.5 * dx, y + 0.5 * dy);
    ctx.lineTo(x - 0.35 * dx + 0.32 * px, y - 0.35 * dy + 0.32 * py);
    ctx.lineTo(x - 0.2 * dx, y - 0.2 * dy);
    ctx.lineTo(x - 0.35 * dx - 0.32 * px, y - 0.35 * dy - 0.32 * py);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    if (resting) { ctx.fillStyle = pal.sheepEdge; ctx.beginPath(); ctx.arc(x, y, 0.1, 0, TAU); ctx.fill(); }
    if (R.drank(step, i)) { ctx.fillStyle = pal.drink; ctx.beginPath(); ctx.arc(x + 0.75 * dx, y + 0.75 * dy, 0.16, 0, TAU); ctx.fill(); }
    else if (R.grazed(step, i)) { ctx.fillStyle = pal.grass1; ctx.beginPath(); ctx.arc(x + 0.75 * dx, y + 0.75 * dy, 0.14, 0, TAU); ctx.fill(); }
    if (opt.ring) {
      const hyd = Math.min(1, R.level(step, i, R.HYD));
      ctx.strokeStyle = levelColour(hyd); ctx.lineWidth = lw(2.5);
      ctx.beginPath(); ctx.arc(x, y, 0.6, -Math.PI / 2, -Math.PI / 2 + hyd * TAU); ctx.stroke();
    }
  } else {
    ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.arc(x, y, 0.3, 0, TAU); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = pal.death; ctx.lineWidth = lw(1.5);
    ctx.beginPath(); ctx.moveTo(x - 0.3, y - 0.3); ctx.lineTo(x + 0.3, y + 0.3); ctx.moveTo(x + 0.3, y - 0.3); ctx.lineTo(x - 0.3, y + 0.3); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (opt.labels && view.s > 9) {
    ctx.fillStyle = pal.text; ctx.font = `${lw(11)}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.globalAlpha = alive ? 0.9 : 0.5;
    ctx.fillText(String(i), x + 0.75 * px - 0.4 * dx, y + 0.75 * py - 0.4 * dy); ctx.globalAlpha = 1;
  }
}

function drawFov(ctx, step, i, alpha, lw, stroke) {
  const x = R.posX(step, i), y = R.posY(step, i), h = R.heading(step, i);
  const th = Math.atan2(-Math.cos(h), Math.sin(h));
  const rng = R.meta.near_range || 8;
  ctx.globalAlpha = alpha; ctx.fillStyle = pal.text;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, rng, th - FOV / 2, th + FOV / 2); ctx.closePath(); ctx.fill();
  if (stroke) { ctx.globalAlpha = 0.5; ctx.strokeStyle = pal.text; ctx.lineWidth = lw(1); ctx.stroke(); }
  ctx.globalAlpha = 1;
}

function drawTrails(ctx, step, lw) {
  if (opt.trail <= 0) return;
  const from = Math.max(0, step - opt.trail);
  if (from >= step) return;
  ctx.lineWidth = lw(1.5); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  const segs = 3;
  for (let i = 0; i < R.S; i++) {
    if (selected >= 0 && i !== selected && opt.follow) continue;
    const colour = opt.fill === 'id' ? ID_COLOURS[i % ID_COLOURS.length] : (i === selected ? pal.select : pal.trail);
    ctx.strokeStyle = colour;
    for (let g = 0; g < segs; g++) {
      const a = from + Math.floor((step - from) * g / segs), b = from + Math.floor((step - from) * (g + 1) / segs);
      if (b <= a) continue;
      ctx.globalAlpha = (i === selected ? 0.9 : 0.55) * ((g + 1) / segs);
      ctx.beginPath();
      let pen = false;
      for (let t = a; t <= b; t++) {
        if (!R.alive(t, i)) { pen = false; continue; }
        const x = R.posX(t, i), y = R.posY(t, i);
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      }
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------- scrub bar
let scrubCols = null;
function resizeScrub() {
  const w = scrub.clientWidth, h = scrub.clientHeight;
  scrub.width = Math.round(w * dpr); scrub.height = Math.round(h * dpr);
  if (!R) return;
  // per pixel column: night share, fog share
  scrubCols = { w, night: new Float32Array(w), fog: new Float32Array(w) };
  for (let x = 0; x < w; x++) {
    const a = Math.floor(x * R.T / w), b = Math.max(a + 1, Math.floor((x + 1) * R.T / w));
    let n = 0, f = 0;
    for (let t = a; t < b && t < R.T; t++) { if (R.night(t)) n++; if (R.fog(t)) f++; }
    scrubCols.night[x] = n / (b - a); scrubCols.fog[x] = f / (b - a);
  }
}
function drawScrub() {
  if (!scrubCols || scrubCols.w !== scrub.clientWidth) resizeScrub();
  const ctx = sctx, w = scrub.clientWidth, h = scrub.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const top = 14, bh = h - top - 4;
  const dayRGB = hexToRGB(pal.dayBand), nightRGB = hexToRGB(pal.nightBand);
  for (let x = 0; x < w; x++) {
    const n = scrubCols.night[x];
    const r = dayRGB[0] + (nightRGB[0] - dayRGB[0]) * n, g = dayRGB[1] + (nightRGB[1] - dayRGB[1]) * n, b = dayRGB[2] + (nightRGB[2] - dayRGB[2]) * n;
    ctx.fillStyle = `rgb(${r | 0} ${g | 0} ${b | 0})`; ctx.fillRect(x, top, 1, bh);
    if (scrubCols.fog[x] > 0) { ctx.fillStyle = pal.fogBand; ctx.globalAlpha = 0.65 * scrubCols.fog[x]; ctx.fillRect(x, top, 1, bh * 0.35); ctx.globalAlpha = 1; }
  }
  // day boundaries and labels
  ctx.fillStyle = pal.dim; ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
  const t0 = R.t(0);
  ctx.fillText(`D${Math.floor(t0 / R.DAY)}`, 2, 0);
  for (let t = 0; t < R.T; t++) {
    if (R.t(t) % R.DAY === 0) {
      const x = t * w / R.T;
      ctx.fillStyle = pal.text; ctx.globalAlpha = 0.5; ctx.fillRect(Math.round(x), top, 1, bh); ctx.globalAlpha = 1;
      ctx.fillStyle = pal.dim; ctx.fillText(`D${Math.floor(R.t(t) / R.DAY)}`, x + 3, 0);
    }
  }
  // event ticks: drinks (bottom third, blue), first approaches (green), deaths (full height, red)
  for (const e of eventsSorted) {
    const x = Math.round(e.step * w / R.T);
    if (e.kind === 'drink') { ctx.fillStyle = pal.drink; ctx.globalAlpha = 0.6; ctx.fillRect(x, top + bh * 0.7, Math.max(1, (e.end - e.step) * w / R.T), bh * 0.3); ctx.globalAlpha = 1; }
    else if (e.kind === 'near') { ctx.fillStyle = pal.near; ctx.fillRect(x, top + bh * 0.4, 2, bh * 0.3); }
  }
  for (const e of eventsSorted) {
    if (e.kind !== 'death') continue;
    const x = Math.round(e.step * w / R.T);
    ctx.fillStyle = pal.death; ctx.fillRect(x - 1, top - 3, 3, bh + 6);
  }
  // playhead
  const x = play.step * w / Math.max(1, R.T - 1);
  ctx.fillStyle = pal.accent; ctx.fillRect(Math.round(x) - 1, top - 4, 3, bh + 8);
  ctx.beginPath(); ctx.moveTo(x - 6, top - 4); ctx.lineTo(x + 6, top - 4); ctx.lineTo(x, top + 3); ctx.closePath(); ctx.fill();
}
function scrubToX(clientX) {
  const rect = scrub.getBoundingClientRect();
  const u = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  setStep(Math.round(u * (R.T - 1)));
}

// ---------------------------------------------------------------- panel
function updateClock(step) {
  const t = R.t(step);
  $('clk-day').textContent = `Day ${Math.floor(t / R.DAY)}`;
  $('clk-time').textContent = clockText(t);
  $('clk-step').textContent = `step ${step.toLocaleString()} / ${(R.T - 1).toLocaleString()}  (t = ${t.toLocaleString()})`;
  $('clk-light').textContent = `${Math.round(R.light(step) * 100)} %`;
  const flags = [];
  if (R.night(step)) flags.push('night');
  if (R.fog(step)) flags.push('fog');
  const alive = countAlive(step);
  flags.push(`${alive} / ${R.S} alive`);
  $('clk-flags').textContent = ' \u00b7 ' + flags.join(' \u00b7 ');
}
function countAlive(step) { let n = 0; for (let i = 0; i < R.S; i++) if (R.alive(step, i)) n++; return n; }

let rows = [];
function buildSheepTable() {
  const tb = $('sheep').querySelector('tbody');
  tb.innerHTML = ''; rows = [];
  for (let i = 0; i < R.S; i++) {
    const tr = document.createElement('tr');
    tr.className = 'row';
    tr.innerHTML = `<td><span class="sw" style="background:${ID_COLOURS[i % ID_COLOURS.length]}"></span>${i}</td><td class="act"></td>` +
      `<td><span class="bar hyd"><i></i></span> <span class="v mono"></span></td><td><span class="bar energy"><i></i></span> <span class="v mono"></span></td>` +
      `<td><span class="bar fat"><i></i></span> <span class="v mono"></span></td><td class="wd mono"></td><td class="st"></td>`;
    tr.onclick = () => { selected = selected === i ? -1 : i; };
    tb.appendChild(tr);
    rows.push({
      tr, act: tr.querySelector('.act'), bars: tr.querySelectorAll('.bar'), vals: tr.querySelectorAll('.v'), wd: tr.querySelector('.wd'), st: tr.querySelector('.st'),
      cache: '',
    });
  }
}
function updateSheepTable(step) {
  for (let i = 0; i < R.S; i++) {
    const r = rows[i];
    const alive = R.alive(step, i);
    const hyd = R.level(step, i, R.HYD), ene = R.level(step, i, R.ENE), fat = R.fat(step, i), wd = R.wdist(step, i);
    const act = R.actionNames[R.action(step, i)];
    const key = `${alive}|${act}|${hyd.toFixed(2)}|${ene.toFixed(2)}|${fat.toFixed(2)}|${wd.toFixed(1)}|${selected === i}|${R.resting(step, i)}|${R.drank(step, i)}`;
    if (key === r.cache) continue;
    r.cache = key;
    r.tr.classList.toggle('sel', selected === i);
    r.tr.classList.toggle('dead', !alive);
    r.act.textContent = alive ? act : 'dead';
    const set = (bar, val, v, cls) => {
      bar.querySelector('i').style.width = `${Math.min(100, val / 1.15 * 100).toFixed(0)}%`;
      bar.classList.toggle('low', cls && val < 0.5 && val >= 0.25); bar.classList.toggle('bad', cls && val < 0.25);
      v.textContent = val.toFixed(2);
    };
    set(r.bars[0], hyd, r.vals[0], true); set(r.bars[1], ene, r.vals[1], true); set(r.bars[2], fat, r.vals[2], false);
    r.wd.textContent = wd.toFixed(1);
    r.st.innerHTML = alive ? (R.drank(step, i) ? '<span class="tag">drinking</span>' : R.grazed(step, i) ? '<span class="tag">bite</span>' : R.resting(step, i) ? '<span class="tag">rest</span>' : '') : '<span class="tag">dead</span>';
  }
}

function buildLegend() {
  const box = $('legend');
  const chips = [];
  for (const [n, c] of Object.entries(ACTION_COLOURS)) chips.push(`<span class="chip"><span class="sw" style="background:${c}"></span>${n}</span>`);
  chips.push('<span class="chip dim">|</span>');
  for (const n of ['tree', 'willow', 'hut', 'salt', 'flower', 'mushroom']) chips.push(`<span class="chip"><span class="sw ${n === 'hut' ? 'sq' : ''}" style="background:var(--${n})"></span>${n}</span>`);
  chips.push('<span class="chip dim">|</span>');
  chips.push(`<span class="chip"><span class="sw" style="background:var(--death)"></span>death</span>`);
  chips.push(`<span class="chip"><span class="sw" style="background:var(--drink)"></span>drink</span>`);
  chips.push(`<span class="chip"><span class="sw" style="background:var(--near)"></span>first near water</span>`);
  box.innerHTML = chips.join('');
}

// ---------------------------------------------------------------- interaction
function setStep(s) { play.step = Math.min(R.T - 1, Math.max(0, s)); play.acc = 0; }
function updatePlayButton() { $('play').innerHTML = play.playing ? '&#x23F8;' : '&#x25B6;'; }
function togglePlay() {
  if (!R) return;
  if (!play.playing && play.step >= R.T - 1) play.step = 0;
  play.playing = !play.playing; play.last = performance.now(); updatePlayButton();
  if (!play.playing) setHash();
}
function setHash() {
  if (!R) return;
  const file = $('run').value;
  history.replaceState(null, '', `#run=${encodeURIComponent(file)}&step=${play.step}`);
}

$('run').onchange = (e) => { if (e.target.value) { location.hash = `run=${encodeURIComponent(e.target.value)}`; loadRun(e.target.value); } };
$('play').onclick = togglePlay;
$('back').onclick = () => setStep(play.step - 1);
$('fwd').onclick = () => setStep(play.step + 1);
$('back100').onclick = () => setStep(play.step - 100);
$('fwd100').onclick = () => setStep(play.step + 100);
$('first').onclick = () => setStep(0);
$('last').onclick = () => setStep(R.T - 1);
$('reset').onclick = () => { opt.follow = false; $('opt-follow').checked = false; resetView(); };
$('prevev').onclick = () => jumpEvent(-1);
$('nextev').onclick = () => jumpEvent(1);
$('speed').onchange = (e) => { play.speed = +e.target.value; };
$('fill').onchange = (e) => { opt.fill = e.target.value; };
$('trail').oninput = (e) => { opt.trail = +e.target.value; $('trailv').textContent = e.target.value; };
for (const [id, key] of [['opt-fov', 'fov'], ['opt-rings', 'rings'], ['opt-labels', 'labels'], ['opt-ring', 'ring'], ['opt-mud', 'mud'], ['opt-tint', 'tint'], ['opt-follow', 'follow']]) {
  $(id).onchange = (e) => { opt[key] = e.target.checked; if (key === 'mud') terrainCanvas = null; };
}

let themeMode = 'auto';
$('theme').onclick = () => {
  themeMode = themeMode === 'auto' ? 'light' : themeMode === 'light' ? 'dark' : 'auto';
  if (themeMode === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', themeMode);
  $('theme').textContent = `theme: ${themeMode}`;
  palette();
};
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => palette());

// zoom and pan
map.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = map.getBoundingClientRect();
  zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015));
  opt.follow = false; $('opt-follow').checked = false;
}, { passive: false });
let drag = null;
map.addEventListener('pointerdown', (e) => {
  map.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy, moved: false };
});
map.addEventListener('pointermove', (e) => {
  const rect = map.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.moved) { view.ox = drag.ox + dx; view.oy = drag.oy + dy; map.classList.add('dragging'); opt.follow = false; $('opt-follow').checked = false; }
    tooltip.hidden = true;
    return;
  }
  updateHover(sx, sy, e.clientX, e.clientY);
});
map.addEventListener('pointerup', (e) => {
  if (drag && !drag.moved && R) {
    const rect = map.getBoundingClientRect();
    const hit = pick(e.clientX - rect.left, e.clientY - rect.top);
    selected = hit && hit.kind === 'sheep' ? (selected === hit.i ? -1 : hit.i) : -1;
  }
  drag = null; map.classList.remove('dragging');
});
map.addEventListener('pointerleave', () => { hover = null; tooltip.hidden = true; });

function pick(sx, sy) {
  if (!R) return null;
  const [wx, wy] = toWorld(sx, sy);
  const step = play.step;
  let best = null, bd = Math.max(0.7, 8 / view.s);
  for (let i = 0; i < R.S; i++) {
    const d = Math.hypot(R.posX(step, i) - wx, R.posY(step, i) - wy);
    if (d < bd) { bd = d; best = { kind: 'sheep', i }; }
  }
  if (best) return best;
  const o = R.world.objects;
  bd = Math.max(0.8, 8 / view.s);
  for (let i = 0; i < o.kind.length; i++) {
    if (!o.present[i]) continue;
    const d = Math.hypot(o.pos[i][1] - wx, o.pos[i][0] - wy);
    if (d < bd) { bd = d; best = { kind: 'object', i }; }
  }
  if (best) return best;
  if (wx >= 0 && wy >= 0 && wx < N && wy < N) return { kind: 'ground', wx, wy };
  return null;
}
function updateHover(sx, sy, cx, cy) {
  hover = pick(sx, sy);
  if (!hover) { tooltip.hidden = true; return; }
  const step = play.step;
  let text;
  if (hover.kind === 'sheep') {
    const i = hover.i;
    const lv = R.meta.drive_names.map((n, k) => `${n} ${R.level(step, i, k).toFixed(2)}`).join('  ');
    text = `sheep ${i}: ${R.alive(step, i) ? R.actionNames[R.action(step, i)] : 'dead'}${R.resting(step, i) ? ' (resting)' : ''}\n${lv}\nfat ${R.fat(step, i).toFixed(2)}  water ${R.wdist(step, i).toFixed(1)} tiles\n` +
      `at (${R.posY(step, i).toFixed(1)}, ${R.posX(step, i).toFixed(1)}) heading ${Math.round(R.heading(step, i) * 180 / Math.PI)} deg\nclick to select`;
  } else if (hover.kind === 'object') {
    const o = R.world.objects, i = hover.i;
    text = `${objectName(o.kind[i])} (variant ${o.variant[i]}) at (${o.pos[i][0].toFixed(1)}, ${o.pos[i][1].toFixed(1)})`;
  } else {
    const nc = R.world.n_cells || 96, res = R.world.grass_res || 2;
    const ci = Math.min(nc - 1, Math.floor(hover.wy * res)), cj = Math.min(nc - 1, Math.floor(hover.wx * res));
    const k = grassSnapshotIndex(step);
    const g = R.arrays.grass[k * nc * nc + ci * nc + cj] / R.sc.grass;
    const rm = R.arrays.regrow_map[ci * nc + cj] / R.sc.regrow_map;
    const wet = R.arrays.water[Math.floor(hover.wy) * N + Math.floor(hover.wx)];
    text = `tile (${hover.wy.toFixed(1)}, ${hover.wx.toFixed(1)})${wet ? '  water' : ''}\ngrass ${g.toFixed(2)} / 3 (snapshot at step ${R.grassSteps[k]})\nregrowth factor ${rm.toFixed(2)}${rm < 1 ? ' (mud)' : ''}\nwater distance ${waterDist(hover.wy, hover.wx).toFixed(1)} tiles`;
  }
  tooltip.textContent = text;
  tooltip.hidden = false;
  const wrap = $('mapwrap').getBoundingClientRect();
  let x = cx - wrap.left + 14, y = cy - wrap.top + 14;
  if (x + tooltip.offsetWidth > wrap.width - 8) x = cx - wrap.left - tooltip.offsetWidth - 10;
  if (y + tooltip.offsetHeight > wrap.height - 8) y = cy - wrap.top - tooltip.offsetHeight - 10;
  tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`;
}
function waterDist(row, col) {
  if (!waterDistField) {
    waterDistField = [];
    const w = R.arrays.water;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (w[r * N + c]) waterDistField.push([r + 0.5, c + 0.5]);
  }
  let best = Infinity;
  for (const [r, c] of waterDistField) best = Math.min(best, Math.hypot(r - row, c - col));
  return best;
}

// scrub bar
let scrubbing = false;
scrub.addEventListener('pointerdown', (e) => { if (!R) return; scrubbing = true; scrub.setPointerCapture(e.pointerId); scrubToX(e.clientX); });
scrub.addEventListener('pointermove', (e) => {
  if (!R) return;
  if (scrubbing) scrubToX(e.clientX);
  const rect = scrub.getBoundingClientRect();
  const u = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  const s = Math.round(u * (R.T - 1));
  scrub.title = `${stepLabel(s)}${R.fog(s) ? ' fog' : ''}${R.night(s) ? ' night' : ''}`;
});
scrub.addEventListener('pointerup', () => { scrubbing = false; setHash(); });

// keys
window.addEventListener('keydown', (e) => {
  if (!R) return;
  const tag = e.target.tagName, type = e.target.type;
  if (tag === 'SELECT' || (tag === 'INPUT' && type !== 'checkbox') || tag === 'TEXTAREA') return;
  if (tag === 'BUTTON' || tag === 'INPUT') e.target.blur();   // a clicked button or checkbox keeps the focus; the keys are the viewer's
  const big = e.shiftKey ? 100 : 1;
  switch (e.key) {
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft': e.preventDefault(); setStep(play.step - big); break;
    case 'ArrowRight': e.preventDefault(); setStep(play.step + big); break;
    case 'Home': setStep(0); break;
    case 'End': setStep(R.T - 1); break;
    case '+': case '=': zoomAt(map.clientWidth / 2, map.clientHeight / 2, 1.25); break;
    case '-': case '_': zoomAt(map.clientWidth / 2, map.clientHeight / 2, 0.8); break;
    case 'f': case 'F': opt.follow = !opt.follow; $('opt-follow').checked = opt.follow; break;
    case 'e': jumpEvent(1); break;
    case 'E': jumpEvent(-1); break;
    case '[': stepSpeed(-1); break;
    case ']': stepSpeed(1); break;
    case 'Escape': selected = -1; break;
    case '0': opt.follow = false; $('opt-follow').checked = false; resetView(); break;
    default:
      if (/^[1-9]$/.test(e.key)) { const i = +e.key; if (i < R.S) selected = selected === i ? -1 : i; break; }
      return;
  }
});
function stepSpeed(dir) {
  const sel = $('speed');
  const i = Math.min(sel.options.length - 1, Math.max(0, sel.selectedIndex + dir));
  sel.selectedIndex = i; play.speed = +sel.value;
}

window.addEventListener('resize', () => { resizeMap(); resizeScrub(); });

// ---------------------------------------------------------------- go
palette();
resizeMap();
loadIndex();
