// ============================================================================
// Публичный API: Vox.load(), model.cloneVoice(), model.synthesize()/stream(), утилиты WAV/воспроизведения.
// ============================================================================
const DEFAULTS = { nfe_step: 32, cfg_strength: 2.0, sway_sampling_coef: -1.0, use_epss: true, speed: 1.0, cross_fade_duration: 0.15, target_rms: 0.1, seed: 1 };

// Голос = mel-спектрограмма образца + его текст + громкость образца. Хранится в памяти страницы; export()/Voice.from() — в JSON.
class Voice {
  constructor(d) { Object.assign(this, d); }
  get seconds() { return this.samples / this.sampleRate; }
  export() {
    const b = new Uint8Array(this.mel.buffer, this.mel.byteOffset, this.mel.byteLength);
    let s = "";
    for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
    return { vox_voice: 1, name: this.name, refText: this.refText, rms: this.rms, samples: this.samples, sampleRate: this.sampleRate, frames: this.frames, melDim: this.melDim, mel: btoa(s) };
  }
  static from(o) {
    if (typeof o === "string") o = JSON.parse(o);
    if (!o || o.vox_voice !== 1) fail("это не голос vox.js (ожидался JSON из voice.export())", "arg");
    const bin = atob(o.mel), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Voice({ name: o.name, refText: o.refText, rms: o.rms, samples: o.samples, sampleRate: o.sampleRate, frames: o.frames, melDim: o.melDim, mel: new Float32Array(u8.buffer) });
  }
}

async function decodeAudio(src) {
  if (src && src.getChannelData) {   // AudioBuffer
    const ch = src.numberOfChannels, n = src.length, out = new Float32Array(n);
    for (let c = 0; c < ch; c++) { const d = src.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / ch; }
    return { pcm: out, sampleRate: src.sampleRate };
  }
  let buf;
  if (typeof src === "string") { const r = await fetch(src); if (!r.ok) fail("не удалось загрузить аудио: HTTP " + r.status, "network"); buf = await r.arrayBuffer(); }
  else if (typeof Blob !== "undefined" && src instanceof Blob) buf = await src.arrayBuffer();
  else if (src instanceof ArrayBuffer) buf = src.slice(0);
  else if (ArrayBuffer.isView(src)) buf = src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength);
  else fail("неизвестный источник звука", "arg");
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) fail("в этом окружении нет AudioContext: передайте Float32Array и sampleRate", "arg");
  const ctx = new Ctx();
  try { return await decodeAudio(await ctx.decodeAudioData(buf)); }
  catch (e) { if (e instanceof VoxError) throw e; fail("браузер не смог декодировать аудио (поддерживаются wav, mp3, ogg, m4a, webm): " + (e && e.message || e), "decode"); }
  finally { if (ctx.close) ctx.close(); }
}

function wavBlob(pcm, sampleRate, opts = {}) {
  const float = !!opts.float, bps = float ? 4 : 2, n = pcm.length;
  const buf = new ArrayBuffer(44 + n * bps), dv = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); dv.setUint32(4, 36 + n * bps, true); str(8, "WAVE"); str(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, float ? 3 : 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * bps, true); dv.setUint16(32, bps, true); dv.setUint16(34, bps * 8, true); str(36, "data"); dv.setUint32(40, n * bps, true);
  for (let i = 0; i < n; i++) {
    if (float) dv.setFloat32(44 + i * 4, pcm[i], true);
    else dv.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32767))), true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function playPcm(pcm, sampleRate, ctx) {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ac = ctx || new Ctx();
  const ab = ac.createBuffer(1, pcm.length, sampleRate);
  ab.copyToChannel(pcm, 0);
  const src = ac.createBufferSource();
  src.buffer = ab; src.connect(ac.destination);
  const ended = new Promise((res) => { src.onended = res; });
  src.start();
  return { stop: () => { try { src.stop(); } catch (e) { /* уже остановлено */ } }, ended, context: ac };
}

// Проигрыватель потока: куски ставятся встык по расписанию, без щелчков.
class StreamPlayer {
  constructor(sampleRate, ctx) {
    const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
    this.ac = ctx || new Ctx();
    this.sr = sampleRate;
    this.t = 0;
    this.nodes = [];
  }
  push(pcm) {
    if (!pcm.length) return;
    const ab = this.ac.createBuffer(1, pcm.length, this.sr);
    ab.copyToChannel(pcm, 0);
    const s = this.ac.createBufferSource();
    s.buffer = ab; s.connect(this.ac.destination);
    const at = Math.max(this.t, this.ac.currentTime + 0.05);
    s.start(at);
    this.t = at + pcm.length / this.sr;
    this.nodes.push(s);
  }
  get ended() { return new Promise((res) => { const w = () => (this.ac.currentTime >= this.t ? res() : setTimeout(w, 50)); w(); }); }
  stop() { for (const s of this.nodes) { try { s.stop(); } catch (e) { /* ок */ } } this.nodes = []; }
}

// Склейка кусков кроссфейдом: в хвост уходят последние cfMax отсчётов, остальное можно отдавать. Те же формулы, что synth_chunk в ядре.
class Splicer {
  constructor(cfMax) { this.cfMax = cfMax; this.tail = new Float32Array(0); this.fadeIn = false; }
  push(wave, first) {
    const cfMax = this.cfMax;
    if (this.fadeIn && cfMax > 0) {   // после паузы между отрезками: нарастание вместо кроссфейда с предыдущим
      const n = Math.min(cfMax, wave.length);
      for (let i = 0; i < n; i++) wave[i] *= i / n;
      this.fadeIn = false;
    }
    if (cfMax === 0) return wave;
    const tail = this.tail;
    let buf;
    if (tail.length === 0 && first) buf = wave;
    else {
      const cfs = Math.min(cfMax, tail.length, wave.length);
      if (cfs === 0) { buf = new Float32Array(tail.length + wave.length); buf.set(tail); buf.set(wave, tail.length); }
      else {
        buf = new Float32Array(tail.length - cfs + wave.length);
        buf.set(tail.subarray(0, tail.length - cfs));
        for (let i = 0; i < cfs; i++) {
          const fo = cfs === 1 ? 1 : 1 - i / (cfs - 1), fi = cfs === 1 ? 0 : i / (cfs - 1);
          buf[tail.length - cfs + i] = tail[tail.length - cfs + i] * fo + wave[i] * fi;
        }
        buf.set(wave.subarray(cfs), tail.length);
      }
    }
    const keep = Math.min(cfMax, buf.length);
    this.tail = buf.slice(buf.length - keep);
    return buf.subarray(0, buf.length - keep);
  }
  // Отдать хвост. fade=true — с затуханием до нуля (отрезок закончился, продолжения пока нет)
  release(fade) {
    const t = this.tail;
    this.tail = new Float32Array(0);
    if (fade && t.length) { const n = t.length; for (let i = 0; i < n; i++) t[i] *= 1 - (i + 1) / n; this.fadeIn = true; }
    return t;
  }
}

class VoxModel {
  constructor(gpu, tm, f5) {
    this.gpu = gpu; this.f5 = f5;
    this.manifest = tm.manifest;
    this.sampleRate = f5.cfg.sample_rate;
    this.generation = Object.assign({}, DEFAULTS, tm.manifest.generation || {});
    const m = tm.manifest.model || {};
    this.info = {
      name: m.name || "", family: m.family || "f5-tts", variant: m.variant || (tm.manifest.f5 && tm.manifest.f5.variant) || "",
      sampleRate: this.sampleRate, weightBytes: f5.wbytes, device: gpu.info, backend: gpu.isCpu ? "cpu" : "webgpu", precision: tm.manifest.precision || {},
      f5: { text_mask_padding: f5.cfg.mask_padding, pe_attn_head: f5.cfg.pe_attn_head || null },
    };
  }

  static async load(src, opts = {}) {
    const t0 = performance.now();
    const bytes = await acquireTmod(src, opts, true);
    const tm = bytes && bytes.qwenSource ? bytes.qwenSource : new Tmod(bytes);
    const fam = (tm.manifest.model || {}).family;
    if (fam && fam !== "f5-tts" && fam !== "qwen3-tts") fail("vox.js умеет семейства f5-tts и qwen3-tts, а в файле: " + fam, "unsupported");
    if (opts.onStatus) opts.onStatus("выбор устройства (WebGPU или процессор)…");
    const gpu = opts.gpu || await createDevice(opts);
    if (opts.onStatus) opts.onStatus((gpu.isCpu ? "подготовка весов для процессора…" : "загрузка весов в GPU…"));
    if (fam === "qwen3-tts") {
      const q = await QwenGpu.create(gpu, tm);
      const qm = new QwenModel(gpu, tm, q);
      qm.info.loadMs = performance.now() - t0;
      if (tm.net) qm.info.net = { requests: tm.net.requests, mib: +(tm.net.bytes / 1048576).toFixed(1), redirects: tm.net.redirects, cachedBlocks: tm.net.cached, netMs: Math.round(tm.net.ms), mibPerSec: tm.net.ms ? +(tm.net.bytes / 1048576 / (tm.net.ms / 1000)).toFixed(1) : 0 };
      return qm;
    }
    if (gpu.precompile) {
      const k = tm.manifest.f5 || {}, mf = tm.manifest.precision || {};
      const wt = mf.quant === "q8" ? [0, 1, 2, 3] : mf.quant === "q4" ? [0, 1, 2, 4] : [0, 1, 2];
      const L = [["attention", "main", { HD: k.dim_head }], ["rope", "main"], ["conv1d", "main"], ["grn", "sumsq"], ["grn", "stats"], ["grn", "apply"], ["ln", "main"]];
      for (const w of wt) L.push(["gemm", "main", { WTYPE: w }]);
      for (const e of ["concat_in", "euler", "euler1", "zero_rows"]) L.push(["ew", e]);
      await gpu.precompile(L);
    }
    const f5 = new F5Gpu(gpu, tm);
    tm.bytes = null;   // веса уже в видеопамяти
    const m = new VoxModel(gpu, tm, f5);
    m.info.loadMs = performance.now() - t0;
    return m;
  }

  // ---- голос ----
  // audio: Float32Array (+ opts.sampleRate), AudioBuffer, Blob/File, ArrayBuffer, URL. refText — что сказано в образце.
  async cloneVoice(audio, refText, opts = {}) {
    if (typeof refText !== "string" || !refText.trim()) fail("нужен текст образца (refText): что произносится в записи", "arg");
    let pcm, sr;
    if (audio instanceof Float32Array) { pcm = audio; sr = opts.sampleRate; if (!sr) fail("для Float32Array укажите opts.sampleRate", "arg"); }
    else ({ pcm, sampleRate: sr } = await decodeAudio(audio));
    const k = this.f5.cfg, target = this.generation.target_rms;
    let a = resampleSinc(pcm, sr, k.sample_rate);
    const maxn = 12 * k.sample_rate;       // как в preprocess_ref_audio_text: не больше 12 с образца
    if (a.length > maxn) a = a.slice(0, maxn); else if (a === pcm) a = a.slice();
    if (a.length < k.mel.n_fft) fail("образец слишком короткий: нужно хотя бы пара секунд речи", "arg");
    let ss = 0;
    for (let i = 0; i < a.length; i++) ss += a[i] * a[i];
    const rms = f32r(Math.sqrt(ss / a.length));
    if (rms < target && rms > 0) { const g = f32r(target / rms); for (let i = 0; i < a.length; i++) a[i] *= g; }
    const { mel, frames } = melLogSpectrogram(k.mel, this.f5.melFb, a);
    return new Voice({ name: opts.name || "", refText: refText.trim(), rms, samples: a.length, sampleRate: k.sample_rate, frames, melDim: k.mel_dim, mel });
  }

  // ---- синтез ----
  _opts(o) {
    const g = this.generation, num = (key, d) => (o[key] != null ? Number(o[key]) : g[key] != null ? Number(g[key]) : d);
    if (o.fix_duration != null) fail("fix_duration пока не поддерживается (используйте speed)", "unsupported");
    const s = {
      nfe: Math.max(1, Math.trunc(num("nfe_step", 32))), cfg: f32r(num("cfg_strength", 2)), sway: f32r(num("sway_sampling_coef", -1)),
      epss: (o.use_epss != null ? !!o.use_epss : g.use_epss != null ? !!g.use_epss : true),
      speed: f32r(num("speed", 1)), crossFade: f32r(num("cross_fade_duration", 0.15)), targetRms: f32r(num("target_rms", 0.1)),
      seed: Math.trunc(num("seed", 1)), maxChars: o.max_chars != null ? Math.trunc(o.max_chars) : 0,
    };
    if (!(s.speed > 0)) fail("speed должен быть положительным", "arg");
    return s;
  }

  // Параметры нарезки для голоса: нормированный текст образца, длина образца в кадрах, предел символов куска, длина кроссфейда.
  _plan(voice, s) {
    const k = this.f5.cfg;
    let rt = voice.refText;
    if (!rt.endsWith(". ") && !rt.endsWith("。")) rt += rt.endsWith(".") ? " " : ". ";
    const refAudioLen = Math.floor(voice.samples / k.mel.hop);
    const refSec = f32r(voice.samples / k.sample_rate);
    let maxChars = s.maxChars;
    if (maxChars <= 0) maxChars = Math.max(1, Math.trunc(blen(rt) / refSec * (22 - refSec) * s.speed));
    const cfMax = Math.floor(f32r(Math.max(0, s.crossFade) * k.sample_rate));
    return { rt, refAudioLen, refSec, maxChars, cfMax };
  }

  // Один кусок текста → звук (mel → Vocos → выравнивание громкости). idx — сквозной номер куска (поток шума).
  async _chunkWave(voice, s, P, gen, idx, opts) {
    const k = this.f5.cfg;
    let ref = P.rt;
    if (ref.charCodeAt(ref.length - 1) < 0x80) ref += " ";
    const ids = this.f5.tok.encode(ref + gen);
    const localSpeed = blen(gen) < 10 ? f32r(0.3) : s.speed;
    let dur = P.refAudioLen + Math.trunc(P.refAudioLen / blen(ref) * blen(gen) / localSpeed);
    dur = Math.max(dur, Math.max(ids.length, voice.frames) + 1);
    dur = Math.min(dur, 65536);
    let noise;
    if (opts.noise) noise = typeof opts.noise === "function" ? opts.noise(idx, dur * k.mel_dim) : opts.noise[idx];
    if (!noise) noise = f5Noise(s.seed, idx, dur * k.mel_dim);
    if (noise.length < dur * k.mel_dim) fail("переданный шум короче нужного для куска " + idx, "arg");
    const mel = await this.f5.sample(voice.mel, voice.frames, ids, dur, { nfe: s.nfe, cfg: s.cfg, sway: s.sway, epss: s.epss }, noise,
      { signal: opts.signal, onStep: opts.onStep ? (a, b) => opts.onStep({ chunk: idx, chunks: opts.chunksTotal || 0, step: a, steps: b }) : null });
    const from = P.refAudioLen;
    const wave = await this.f5.vocode(mel.subarray(from * k.mel_dim), dur - from);
    if (voice.rms < s.targetRms && voice.rms > 0) { const g = f32r(voice.rms / s.targetRms); for (let i = 0; i < wave.length; i++) wave[i] *= g; }
    return wave;
  }

  async *stream(text, opts = {}) {
    const voice = opts.voice;
    if (!(voice instanceof Voice)) fail("передайте голос: opts.voice = await model.cloneVoice(...)", "arg");
    if (voice.melDim !== this.f5.cfg.mel_dim) fail("голос создан для другой модели", "arg");
    if (typeof text !== "string" || !text.trim()) fail("пустой текст", "arg");
    const s = this._opts(opts), k = this.f5.cfg;
    const P = this._plan(voice, s);
    const chunks = chunkText(text, P.maxChars);
    const sp = new Splicer(P.cfMax);
    const t0 = performance.now();
    this.f5.timing = { dit_ms: 0, vocos_ms: 0, text_ms: 0 };   // статистика — за этот запуск
    let produced = 0;
    for (let idx = 0; idx < chunks.length; idx++) {
      if (opts.signal && opts.signal.aborted) fail("синтез отменён", "aborted");
      const wave = await this._chunkWave(voice, s, P, chunks[idx], idx, Object.assign({}, opts, { chunksTotal: chunks.length }));
      const piece = sp.push(wave, idx === 0);
      produced += piece.length;
      if (piece.length) yield piece;
    }
    const tail = sp.release(false);
    if (tail.length) { produced += tail.length; yield tail; }
    this.lastRun = { audioSeconds: produced / k.sample_rate, wallMs: performance.now() - t0, chunks: chunks.length, timing: Object.assign({}, this.f5.timing) };
  }

  // Весь звук целиком. opts: voice, seed, nfe_step, cfg_strength, sway_sampling_coef, speed, cross_fade_duration, max_chars,
  // onChunk(Float32Array), onStep({chunk,chunks,step,steps}), signal (AbortSignal).
  async synthesize(text, opts = {}) {
    const parts = [];
    let n = 0;
    for await (const p of this.stream(text, opts)) { parts.push(p); n += p.length; if (opts.onChunk) opts.onChunk(p); }
    const pcm = new Float32Array(n);
    let o = 0;
    for (const p of parts) { pcm.set(p, o); o += p.length; }
    return { pcm, sampleRate: this.sampleRate, seconds: n / this.sampleRate, stats: this.lastRun };
  }

  // log-mel спектрограмма готового звука (для отрисовки): { mel: Float32Array [frames][bands], frames, bands }
  spectrogram(pcm) {
    const k = this.f5.cfg;
    const r = melLogSpectrogram(k.mel, this.f5.melFb, pcm);
    return { mel: r.mel, frames: r.frames, bands: k.mel_dim };
  }

  dispose() { this.f5.dispose(); try { this.gpu.device.destroy(); } catch (e) { /* ок */ } }
}
