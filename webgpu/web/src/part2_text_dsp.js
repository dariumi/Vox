// ============================================================================
// Текст: токенизатор F5 (латиница посимвольно, китайский через таблицу пиньиня из .tmod), нарезка на куски как в f5_tts.
// ============================================================================
const utf8 = new TextEncoder();
const blen = (s) => utf8.encode(s).length;

class F5Tokenizer {
  constructor(tok) {
    this.pinyin = (tok.type || "pinyin") === "pinyin";
    this.vocab = new Map();
    tok.vocab.forEach((s, i) => { if (!this.vocab.has(s)) this.vocab.set(s, i); });
    this.pmap = new Map();
    if (tok.pinyin) for (const k of Object.keys(tok.pinyin)) this.pmap.set(k.codePointAt(0), tok.pinyin[k]);
  }
  symbols(text) {
    if (!this.pinyin) return Array.from(text);
    const cps = [];
    for (const ch of text) {
      let c = ch.codePointAt(0);
      if (c === 0x3B) c = 0x2C;                        // ; → ,
      else if (c === 0x201C || c === 0x201D) c = 0x22; // “ ” → "
      else if (c === 0x2018 || c === 0x2019) c = 0x27; // ‘ ’ → '
      cps.push(c);
    }
    const isAl = (c) => (c >= 48 && c <= 57) || (c >= 97 && c <= 122) || (c >= 65 && c <= 90);
    const isDg = (c) => c >= 48 && c <= 57;
    const out = [], n = cps.length;
    let i = 0;
    while (i < n) {
      const c = cps[i];
      if (isAl(c)) {   // сегмент [A-Za-z0-9]+(\.\d+)?%? — одно слово для jieba
        let j = i;
        while (j < n && isAl(cps[j])) j++;
        if (j + 1 < n && cps[j] === 46 && isDg(cps[j + 1])) { j++; while (j < n && isDg(cps[j])) j++; }
        if (j < n && cps[j] === 37) j++;
        if (j - i > 1 && out.length) {
          const last = out[out.length - 1];
          if (!(last === " " || last === ":" || last === "'" || last === '"')) out.push(" ");
        }
        for (let k = i; k < j; k++) out.push(String.fromCharCode(cps[k]));
        i = j;
        continue;
      }
      if (c >= 0x3100 && c <= 0x9FFF) {
        out.push(" ");
        const p = this.pmap.get(c);
        out.push(p === undefined ? String.fromCodePoint(c) : p);
      } else out.push(String.fromCodePoint(c));
      i++;
    }
    return out;
  }
  encode(text) { return this.symbols(text).map((s) => (this.vocab.has(s) ? this.vocab.get(s) : 0)); }
}

const FULLWIDTH_STOPS = new Set(["；", "：", "，", "。", "！", "？"]);
function splitSentences(text) {
  const out = [];
  let cur = "";
  const cs = Array.from(text);
  const isWs = (c) => c === " " || c === "\t" || c === "\n" || c === "\r" || c === "\f" || c === "\v";
  for (let i = 0; i < cs.length; i++) {
    const c = cs[i];
    cur += c;
    let boundary = false;
    if (c.length === 1 && c.charCodeAt(0) < 0x80) {
      if (";:,.!?".includes(c)) {
        let j = i + 1;
        while (j < cs.length && isWs(cs[j])) j++;
        if (j > i + 1) { boundary = true; i = j - 1; }
      }
    } else if (FULLWIDTH_STOPS.has(c)) boundary = true;
    if (boundary) { out.push(cur); cur = ""; }
  }
  if (cur) out.push(cur);
  return out;
}
const pyStrip = (s) => s.replace(/^[ \t\n\r\f\v]+|[ \t\n\r\f\v]+$/g, "");
function chunkText(text, maxChars) {
  const chunks = [];
  let cur = "", curLen = 0;
  for (const sent of splitSentences(text)) {
    if (!sent) continue;
    const sl = blen(sent);
    const lastAscii = sent.charCodeAt(sent.length - 1) < 0x80;
    const add = lastAscii ? sent + " " : sent;
    if (curLen + sl <= maxChars) { cur += add; curLen += blen(add); }
    else {
      if (cur) chunks.push(pyStrip(cur));
      cur = add; curLen = blen(add);
    }
  }
  if (cur) chunks.push(pyStrip(cur));
  return chunks;
}

// ============================================================================
// DSP: БПФ, mel-спектрограмма образца, обратное STFT для Vocos (в double, как в ядре)
// ============================================================================
class FFT {
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      this.rev[i] = j;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { this.cos[i] = Math.cos(2 * Math.PI * i / n); this.sin[i] = Math.sin(2 * Math.PI * i / n); }
  }
  // на месте; sign = -1 прямое, +1 обратное (без нормировки)
  run(re, im, sign) {
    const n = this.n, rev = this.rev;
    for (let i = 1; i < n; i++) { const j = rev[i]; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step], wi = sign * this.sin[k * step];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }
}
const hannPeriodic = (n) => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; };

// log-mel как в F5/Vocos: reflect-дополнение, периодическое окно Ханна, |STFT|, банк фильтров, log(max(x, 1e-5)).
function melLogSpectrogram(cfg, fb, pcm) {
  const n = pcm.length, nfft = cfg.n_fft, hop = cfg.hop, nb = (nfft >> 1) + 1, pad = nfft >> 1, nm = cfg.n_mels;
  if (n <= pad) fail("образец слишком короткий для mel-спектрограммы", "arg");
  const frames = 1 + Math.floor(n / hop);
  const xp = new Float32Array(n + 2 * pad);
  for (let i = -pad; i < n + pad; i++) {
    let j = i < 0 ? -i : i;
    if (j >= n) j = 2 * (n - 1) - j;
    xp[i + pad] = pcm[j];
  }
  const win = new Float64Array(nfft);
  const h = hannPeriodic(cfg.win), off = (nfft - cfg.win) >> 1;
  for (let i = 0; i < cfg.win; i++) win[off + i] = h[i];
  const fft = new FFT(nfft);
  const re = new Float64Array(nfft), im = new Float64Array(nfft), mag = new Float64Array(nb);
  const out = new Float32Array(frames * nm);
  for (let t = 0; t < frames; t++) {
    const base = t * hop;
    for (let i = 0; i < nfft; i++) { re[i] = xp[base + i] * win[i]; im[i] = 0; }
    fft.run(re, im, -1);
    for (let k = 0; k < nb; k++) mag[k] = f32r(Math.hypot(re[k], im[k]));
    for (let m = 0; m < nm; m++) {
      let acc = 0;
      const o = m * nb;
      for (let k = 0; k < nb; k++) acc += fb[o + k] * mag[k];
      out[t * nm + m] = Math.log(Math.max(acc, 1e-5));
    }
  }
  return { mel: out, frames };
}

// Выход головы Vocos: [T][2*nb] = (log|X|, фаза) → звук ("same" padding, как ISTFT в Vocos).
function istftSame(head, T, nfft, hop) {
  const nb = (nfft >> 1) + 1, pad = (nfft - hop) >> 1, total = (T - 1) * hop + nfft;
  const win = hannPeriodic(nfft), fft = new FFT(nfft);
  const y = new Float64Array(total), env = new Float64Array(total);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < T; t++) {
    const row = t * 2 * nb;
    for (let k = 0; k < nb; k++) {
      const m = Math.min(Math.exp(head[row + k]), 1e2), p = head[row + nb + k];
      let vr = m * Math.cos(p), vi = m * Math.sin(p);
      if (k === 0 || k === nb - 1) vi = 0;
      re[k] = vr; im[k] = vi;
      if (k > 0 && k < nb - 1) { re[nfft - k] = vr; im[nfft - k] = -vi; }
    }
    fft.run(re, im, +1);
    const o = t * hop;
    for (let i = 0; i < nfft; i++) { y[o + i] += f32r(re[i] / nfft) * win[i]; env[o + i] += win[i] * win[i]; }
  }
  const out = new Float32Array(total - 2 * pad);
  for (let i = 0; i < out.length; i++) out[i] = y[i + pad] / env[i + pad];
  return out;
}
