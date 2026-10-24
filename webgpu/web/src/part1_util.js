// ============================================================================
// vox.js — универсальный TTS-движок vox в браузере (WebGPU). Семейство F5-TTS (DiT + Vocos).
// Один файл, без зависимостей:  <script src="vox.js"></script>  →  window.Vox
// Эта часть: утилиты, формат .tmod, форматы весов, Philox-шум, расписание шагов.
// ============================================================================

const VOX_VERSION = "0.16.0";
// Папка, откуда загружен vox.js: рядом лежат pinyin.json и прочие ресурсы. Меняется через Vox.config({ baseUrl }).
let VOX_BASE_URL = "";
try { if (typeof document !== "undefined" && document.currentScript && document.currentScript.src) VOX_BASE_URL = document.currentScript.src.replace(/[^/]*$/, ""); } catch (e) { /* ок */ }

class VoxError extends Error {
  constructor(message, code) { super(message); this.name = "VoxError"; this.code = code || "error"; }
}
const fail = (msg, code) => { throw new VoxError(msg, code); };

// ---------- числа ----------
const f32r = Math.fround;
const _cvt = new DataView(new ArrayBuffer(8));
function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}
function bf16ToF32(h) { _cvt.setUint32(0, (h << 16) >>> 0); return _cvt.getFloat32(0); }

function bessel_i0(x) {
  let sum = 1, term = 1;
  const q = x * x / 4;
  for (let k = 1; k < 200; k++) { term *= q / (k * k); sum += term; if (term < sum * 1e-17) break; }
  return sum;
}

// Окно Кайзера + sinc, как resample_sinc в ядре vox (одинаковый результат с C++ в пределах float).
function resampleSinc(x, srFrom, srTo) {
  const n = x.length;
  if (srFrom === srTo || n === 0) return x;
  const ratio = srTo / srFrom;
  const cutoff = Math.min(1, ratio) * 0.97;
  const zc = 32, half = zc / cutoff, beta = 9.0, i0b = bessel_i0(beta);
  const nout = Math.ceil(n * ratio);
  const y = new Float32Array(nout);
  for (let m = 0; m < nout; m++) {
    const t = m / ratio;
    const lo = Math.ceil(t - half), hi = Math.floor(t + half);
    let acc = 0, wsum = 0;
    for (let k = lo; k <= hi; k++) {
      const d = t - k, u = d / half;
      if (u <= -1 || u >= 1) continue;
      const win = bessel_i0(beta * Math.sqrt(1 - u * u)) / i0b;
      const arg = Math.PI * d * cutoff;
      const s = Math.abs(arg) < 1e-12 ? 1 : Math.sin(arg) / arg;
      const h = cutoff * s * win;
      wsum += h;
      if (k >= 0 && k < n) acc += h * x[k];
    }
    y[m] = wsum !== 0 ? acc / wsum : 0;
  }
  return y;
}

// ---------- Philox4x32-10 (тот же генератор, что в ядре: шум не зависит от устройства) ----------
function mulhilo(a, b) {
  const aL = a & 0xffff, aH = a >>> 16, bL = b & 0xffff, bH = b >>> 16;
  const ll = aL * bL, lh = aL * bH, hl = aH * bL, hh = aH * bH;
  const mid = (ll >>> 16) + (lh & 0xffff) + (hl & 0xffff);
  return [(hh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0, (((mid & 0xffff) << 16) | (ll & 0xffff)) >>> 0];
}
function philoxBlock(c0, c1, c2, c3, k0, k1) {
  for (let i = 0; i < 10; i++) {
    const [h0, l0] = mulhilo(0xD2511F53, c0);
    const [h1, l1] = mulhilo(0xCD9E8D57, c2);
    const n0 = (h1 ^ c1 ^ k0) >>> 0, n2 = (h0 ^ c3 ^ k1) >>> 0;
    c0 = n0; c1 = l1; c2 = n2; c3 = l0;
    if (i < 9) { k0 = (k0 + 0x9E3779B9) >>> 0; k1 = (k1 + 0xBB67AE85) >>> 0; }
  }
  return c0;
}
// Нормальный шум N(0,1) для куска `chunk`: Box–Muller над счётчиковым Philox (как F5Model::noise).
function f5Noise(seed, chunk, n) {
  const k0 = seed >>> 0, k1 = Math.floor(seed / 4294967296) >>> 0;
  const stream = (0xF5000000 + chunk) >>> 0;
  const out = new Float32Array(n);
  const inv = 1 / 16777216;
  for (let p = 0; p < ((n + 1) >> 1); p++) {
    const u1 = (philoxBlock(p >>> 0, Math.floor(p / 4294967296) >>> 0, stream, 0, k0, k1) >>> 8) * inv;
    const u2 = (philoxBlock(p >>> 0, Math.floor(p / 4294967296) >>> 0, stream, 1, k0, k1) >>> 8) * inv;
    const r = Math.sqrt(-2 * Math.log(1 - u1)), a = 2 * Math.PI * u2;
    out[2 * p] = r * Math.cos(a);
    if (2 * p + 1 < n) out[2 * p + 1] = r * Math.sin(a);
  }
  return out;
}

// Сетка времени CFM: linspace или EPSS, затем sway.
const EPSS = {
  5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
  10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], 12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
  16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
};
function f5Timesteps(nfe, sway, epss) {
  const n = Math.max(1, nfe | 0);
  let t;
  if (epss && EPSS[n]) t = EPSS[n].map((v) => v / 32);
  else { t = []; for (let i = 0; i <= n; i++) t.push(i / n); }
  return t.map((v) => f32r(v + sway * (Math.cos(Math.PI / 2 * v) - 1 + v)));
}

// ---------- .tmod ----------
class Tmod {
  constructor(bytes) {
    if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
    if (bytes.length < 64) fail("файл слишком мал для .tmod", "format");
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== "VOXM") fail("это не файл .tmod (нет сигнатуры VOXM)", "format");
    if (dv.getUint32(4, true) !== 1) fail("версия формата .tmod не поддерживается этой сборкой vox.js", "format");
    const u64 = (o) => Number(dv.getBigUint64(o, true));
    const mo = u64(16), ms = u64(24), dof = u64(32), dsz = u64(40);
    if (mo + ms > bytes.length || dof + dsz > bytes.length) fail("файл .tmod обрезан (не докачался?)", "format");
    this.manifest = JSON.parse(new TextDecoder().decode(bytes.subarray(mo, mo + ms)));
    this.bytes = bytes;
    this.dataOffset = dof;
    this.table = this.manifest.blobs || {};
  }
  has(name) { return Object.prototype.hasOwnProperty.call(this.table, name); }
  info(name) {
    const b = this.table[name];
    if (!b) fail("в .tmod нет тензора: " + name, "format");
    return b;
  }
  // сырые байты тензора (вид, без копии)
  raw(name) {
    const b = this.info(name);
    return this.bytes.subarray(this.dataOffset + b.offset, this.dataOffset + b.offset + b.nbytes);
  }
  f32(name) {
    const b = this.info(name);
    if (b.dtype !== "F32") fail("тензор должен быть F32: " + name, "format");
    const r = this.raw(name);
    if ((r.byteOffset & 3) === 0) return new Float32Array(r.buffer, r.byteOffset, r.byteLength >> 2);
    return new Float32Array(r.slice().buffer);
  }
  shape(name) { return this.info(name).shape; }
}

// Матрица весов [rows][cols] в одном из форматов vox (для CPU-стороны: таблица токенов, MLP времени).
function matInfo(tm, name) {
  const b = tm.info(name);
  const rows = b.shape[0];
  let cols = b.shape.length > 1 ? b.shape[1] : 1;
  let wtype;
  switch (b.dtype) {
    case "F32": wtype = 0; break;
    case "F16": wtype = 1; break;
    case "BF16": wtype = 2; break;
    case "I8": wtype = 3; break;
    case "U8": wtype = 4; cols *= 2; break;
    default: fail("неподдерживаемый формат весов у " + name + ": " + b.dtype, "format");
  }
  return { name, rows, cols, wtype, scale: wtype >= 3 ? tm.f32(name + ".scale") : null, raw: null };
}
function matRow(tm, m, r, out) {
  const raw = m.raw || tm.raw(m.name);
  const C = m.cols;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  switch (m.wtype) {
    case 0: for (let c = 0; c < C; c++) out[c] = dv.getFloat32((r * C + c) * 4, true); break;
    case 1: for (let c = 0; c < C; c++) out[c] = f16ToF32(dv.getUint16((r * C + c) * 2, true)); break;
    case 2: for (let c = 0; c < C; c++) out[c] = bf16ToF32(dv.getUint16((r * C + c) * 2, true)); break;
    case 3: for (let c = 0; c < C; c++) out[c] = dv.getInt8(r * C + c) * m.scale[r * (C >> 5) + (c >> 5)]; break;
    default:
      for (let c = 0; c < C; c++) {
        const blk = c >> 5, j = c & 31;
        const byte = raw[r * (C >> 1) + blk * 16 + (j & 15)];
        const u = (byte >> (j >> 4 ? 4 : 0)) & 15;
        out[c] = (u - 8) * m.scale[r * (C >> 5) + blk];
      }
  }
  return out;
}
function matFloat(tm, m) {
  const out = new Float32Array(m.rows * m.cols), tmp = new Float32Array(m.cols);
  for (let r = 0; r < m.rows; r++) { matRow(tm, m, r, tmp); out.set(tmp, r * m.cols); }
  return out;
}

// ---------- загрузка файла с прогрессом ----------
async function fetchBytes(src, onProgress) {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (typeof Blob !== "undefined" && src instanceof Blob) {
    const buf = new Uint8Array(await src.arrayBuffer());
    if (onProgress) onProgress(1, buf.length, buf.length);
    return buf;
  }
  const resp = typeof Response !== "undefined" && src instanceof Response ? src : await fetch(src);
  if (!resp.ok) fail("не удалось загрузить модель: HTTP " + resp.status + " " + resp.statusText, "network");
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body || !resp.body.getReader) {
    const buf = new Uint8Array(await resp.arrayBuffer());
    if (onProgress) onProgress(1, buf.length, buf.length);
    return buf;
  }
  const reader = resp.body.getReader();
  let got = 0, whole = total ? new Uint8Array(total) : null;
  const parts = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (whole && got + value.length <= whole.length) whole.set(value, got); else { if (whole) { parts.push(whole.subarray(0, got)); whole = null; } parts.push(value); }
    got += value.length;
    if (onProgress) onProgress(total ? got / total : 0, got, total);
  }
  if (whole && got === whole.length) return whole;
  if (whole) parts.push(whole.subarray(0, got));
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
