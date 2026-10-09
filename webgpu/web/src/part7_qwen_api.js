// ============================================================================
// Qwen3-TTS в браузере: голос (.tvoice / готовый диктор / описание), сессия генерации (порт engine.cpp + backend_qwen3.cpp),
// детерминированный сэмплинг (Philox), потоковая выдача и адаптер для двустороннего стриминга.
// ============================================================================
const QWEN_DEFAULTS = { temperature: 0.9, top_p: 1.0, top_k: 50, repetition_penalty: 1.05, sub_temperature: 0.9, sub_top_p: 1.0, sub_top_k: 50, max_frames: 2048, chunk_frames: 4 };

const QWEN_ISO = { zh: "chinese", en: "english", ru: "russian", de: "german", fr: "french", es: "spanish", it: "italian", pt: "portuguese", ja: "japanese", ko: "korean" };
function qwenLanguage(l) { l = String(l || "auto").toLowerCase(); return QWEN_ISO[l] || l; }

// Голос Qwen3-TTS: клон (icl — с кодами образца, xvector_only — только вектор диктора), готовый диктор или описание голоса.
class QVoice {
  constructor(d) { Object.assign(this, { mode: "icl", name: "", speaker: "", instruct: "", refText: "", language: "auto", sampling: {}, speakerEmb: new Float32Array(0), refCodes: null, refFrames: 0, refCodebooks: 0, refTextTokens: null }, d); }
  // .tvoice (safetensors, создаётся `vox voice create` или Python) → голос
  static from(bytes) {
    if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
    const t = parseSafetensors(bytes), meta = t.metadata || {};
    if (!meta.voice || !meta.format) fail("это не файл .tvoice (нет метаданных)", "format");
    if (String(meta.format)[0] !== "1") fail("формат .tvoice " + meta.format + " не поддерживается этой сборкой", "format");
    const j = JSON.parse(meta.voice);
    const mode = j.mode === "xvector_only" ? "xvector_only" : j.mode === "speaker" ? "speaker" : "icl";
    const take = (name, Ctor, dtype) => {
      const e = t.get(name);
      if (!e) fail(".tvoice: нет тензора " + name, "format");
      if (e.dtype !== dtype) fail(".tvoice: " + name + " должен быть " + dtype, "format");
      const c = e.bytes.slice();
      return { a: new Ctor(c.buffer, 0, c.byteLength / Ctor.BYTES_PER_ELEMENT), shape: e.shape };
    };
    const v = new QVoice({ mode, name: j.name || "voice", speaker: j.speaker || "", instruct: j.instruct || "", refText: j.ref_text || "", language: j.language || "auto",
      sampling: j.sampling || {}, model: j.model || {}, speakerEmb: take("speaker_emb", Float32Array, "F32").a });
    if (mode === "icl") {
      const c = take("ref_codes", Int16Array, "I16");
      v.refCodes = c.a; v.refFrames = c.shape[0]; v.refCodebooks = c.shape[1];
      v.refTextTokens = take("ref_text_tokens", Int32Array, "I32").a;
    }
    return v;
  }
  // Байты .tvoice (safetensors; тот же формат, что у `vox voice create`) — голос можно сохранить и открыть в любой сборке vox
  toTvoice() {
    const te = new TextEncoder();
    const meta = { format: "1.0", voice: JSON.stringify({
      format: "1.0", name: this.name || "voice", mode: this.mode, ref_text: this.refText || "", speaker: this.speaker || "", instruct: this.instruct || "", language: this.language || "auto",
      model: this.model || { family: "qwen3-tts", variant: "", weights_sha256: "" }, sampling: this.sampling || {}, kv_fingerprints: [], speaker_emb_dim: this.speakerEmb.length,
      ref_frames: this.refFrames || 0, ref_codebooks: this.refCodebooks || 0, has_ref_audio: false, features: [], created: new Date().toISOString().slice(0, 10) }) };
    const ts = [["speaker_emb", "F32", [this.speakerEmb.length], new Uint8Array(this.speakerEmb.buffer, this.speakerEmb.byteOffset, this.speakerEmb.byteLength)]];
    if (this.mode === "icl" && this.refFrames > 0) {
      ts.push(["ref_codes", "I16", [this.refFrames, this.refCodebooks], new Uint8Array(this.refCodes.buffer, this.refCodes.byteOffset, this.refCodes.byteLength)]);
      ts.push(["ref_text_tokens", "I32", [this.refTextTokens.length], new Uint8Array(this.refTextTokens.buffer, this.refTextTokens.byteOffset, this.refTextTokens.byteLength)]);
    }
    const hdr = { __metadata__: meta };
    let off = 0;
    for (const [n, dt, shape, b] of ts) { hdr[n] = { dtype: dt, shape, data_offsets: [off, off + b.length] }; off += b.length; }
    let hb = te.encode(JSON.stringify(hdr));
    const padn = (8 - (hb.length % 8)) % 8;
    if (padn) { const t = new Uint8Array(hb.length + padn); t.set(hb); t.fill(32, hb.length); hb = t; }
    const out = new Uint8Array(8 + hb.length + off);
    new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
    out.set(hb, 8);
    let o = 8 + hb.length;
    for (const [, , , b] of ts) { out.set(b, o); o += b.length; }
    return out;
  }
}

// ---------- сэмплинг (порт sampling.hpp) ----------
function qUniform(k0, k1, frame, cb) {
  return (philoxBlock(frame >>> 0, Math.floor(frame / 4294967296) >>> 0, cb >>> 0, 0, k0, k1) >>> 8) * (1 / 16777216);
}
// logits: Float32Array (может быть изменён: штраф за повтор). p: { temperature, top_p, top_k, repetition_penalty } (float32-значения).
function qSample(logits, p, u, history) {
  const n = logits.length;
  if (!n) return -1;
  if (history && p.repetition_penalty !== 1) {
    const seen = new Uint8Array(n);
    for (const t of history) {
      if (t >= 0 && t < n && !seen[t]) { seen[t] = 1; const l = logits[t]; logits[t] = l > 0 ? l / p.repetition_penalty : l * p.repetition_penalty; }
    }
  }
  if (p.temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const k = p.top_k > 0 && p.top_k < n ? p.top_k : n;
  // k лучших по (логит убыв., индекс возр.)
  let idx;
  if (k === n) { idx = Array.from({ length: n }, (_, i) => i); idx.sort((a, b) => (logits[a] > logits[b] ? -1 : logits[a] < logits[b] ? 1 : a - b)); }
  else {
    idx = [];
    const better = (a, b) => logits[a] > logits[b] || (logits[a] === logits[b] && a < b);
    for (let i = 0; i < n; i++) {
      if (idx.length === k && !better(i, idx[k - 1])) continue;
      let lo = 0, hi = idx.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (better(idx[mid], i)) lo = mid + 1; else hi = mid; }
      idx.splice(lo, 0, i);
      if (idx.length > k) idx.pop();
    }
  }
  const invT = 1 / p.temperature, mx = logits[idx[0]];
  const probs = new Float64Array(idx.length);
  let sum = 0;
  for (let i = 0; i < idx.length; i++) { probs[i] = Math.exp((logits[idx[i]] - mx) * invT); sum += probs[i]; }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  let keep = probs.length;
  if (p.top_p > 0 && p.top_p < 1) {
    let c = 0;
    for (let i = 0; i < probs.length; i++) { c += probs[i]; if (c >= p.top_p) { keep = i + 1; break; } }
  }
  let total = 0;
  for (let i = 0; i < keep; i++) total += probs[i];
  const target = u * total;
  let acc = 0;
  for (let i = 0; i < keep; i++) { acc += probs[i]; if (target < acc) return idx[i]; }
  return idx[keep - 1];
}

// ---------- сессия: один текст → поток кадров → поток звука ----------
class QwenSession {
  constructor(model, voice, opts = {}) {
    const q = model.q, tc = q.tc, g = model.generation;
    this.model = model; this.q = q; this.voice = voice; this.opts = opts;
    const f = Math.fround;
    const pick = (key, ...alts) => { for (const a of [opts[key], voice.sampling && voice.sampling[key], g[key]]) if (a != null) return Number(a); return QWEN_DEFAULTS[key]; };
    this.talker = { temperature: f(pick("temperature")), top_p: f(pick("top_p")), top_k: Math.trunc(pick("top_k")), repetition_penalty: f(pick("repetition_penalty")) };
    this.sub = { temperature: f(pick("sub_temperature")), top_p: f(pick("sub_top_p")), top_k: Math.trunc(pick("sub_top_k")), repetition_penalty: 1 };
    let seed = Math.trunc(opts.seed != null ? opts.seed : (voice.sampling && voice.sampling.seed) || 0);
    if (!seed) seed = 1;
    this.k0 = seed >>> 0; this.k1 = Math.floor(seed / 4294967296) >>> 0;
    this.chunk = Math.max(1, Math.trunc(pick("chunk_frames")));
    this.maxFrames = Math.max(1, Math.trunc(pick("max_frames")));
    this.signal = opts.signal || null;
    this.noEos = !!opts.noEos;
    this.minEos = 2;
    this.suppressFrom = Math.max(0, tc.vocab - 1024); this.suppressTo = tc.vocab;
    this.textStream = []; this.textN = 0;
    this.prefilled = false; this.framesDone = 0; this.frame = 0;
    this.pending = null; this.codes = new Int32Array(tc.groups); this.flat = []; this.buffered = 0; this.history0 = [];
    this.finished = false; this.flushed = false; this.textEnded = false;
    this.ts = null; this.cs = null; this.warmed = false;
    this.stats = { prefill_ms: 0, talker_ms: 0, predictor_ms: 0, codec_ms: 0, frames: 0, prefill_rows: 0 };
    this.ctxFrames = Math.trunc(opts.decoder_context_frames || g.decoder_context_frames || 50);
  }

  _abortCheck() { if (this.signal && this.signal.aborted) fail("синтез отменён", "aborted"); }

  // begin(): префикс (роль + кодек-строки [+ ICL])
  async init() {
    const q = this.q, tc = q.tc, D = tc.hidden, v = this.voice, model = this.model;
    const named = v.mode === "speaker";
    if (!named && v.speakerEmb.length !== D) fail("вектор диктора в голосе имеет размер " + v.speakerEmb.length + ", а модель ждёт " + D, "format");
    const sp = await q.textProject([q.ids.tts_bos, q.ids.tts_eos, q.ids.tts_pad]);
    this.bos = sp.slice(0, D); this.eos = sp.slice(D, 2 * D); this.pad = sp.slice(2 * D, 3 * D);
    let lang = qwenLanguage(this.opts.language || v.language), langId = -1;
    if (lang !== "auto") {
      if (tc.lang[lang] === undefined) fail("язык '" + lang + "' не поддерживается этой моделью (есть: " + Object.keys(tc.lang).join(", ") + ")", "arg");
      langId = tc.lang[lang];
    }
    let spkRow = -1;
    if (named && v.speaker) {
      const key = v.speaker.toLowerCase();
      if (tc.spk_id[key] === undefined) fail("диктора '" + v.speaker + "' нет в модели", "arg");
      spkRow = tc.spk_id[key];
      const dl = tc.spk_dialect[key];
      if ((lang === "chinese" || lang === "auto") && typeof dl === "string") {
        if (tc.lang[dl] === undefined) fail("диалект '" + dl + "' диктора '" + v.speaker + "' не имеет id языка", "format");
        langId = tc.lang[dl];
      }
    }
    const prefix = langId < 0 ? [tc.nothink, tc.think_bos, tc.think_eos] : [tc.think, tc.think_bos, langId, tc.think_eos];
    const hasSpk = !named || spkRow >= 0;
    const C = prefix.length + (hasSpk ? 3 : 2);
    const cie = new Float32Array(C * D);
    let r = 0;
    for (const id of prefix) q.codecRow(id, cie.subarray(r++ * D, r * D));
    if (named) { if (spkRow >= 0) q.codecRow(spkRow, cie.subarray(r++ * D, (r) * D)); }
    else { cie.set(v.speakerEmb, r * D); r++; }
    q.codecRow(tc.pad, cie.subarray(r++ * D, r * D));
    q.codecRow(tc.bos, cie.subarray(r++ * D, r * D));
    // инструкция
    let ins = [];
    if (named && v.instruct) ins = model.tok.encode("<|im_start|>user\n" + v.instruct + "<|im_end|>\n");
    const I = ins.length;
    const role = model.tok.encode("<|im_start|>assistant\n");
    if (role.length !== 3) fail("префикс роли должен быть из 3 токенов", "format");
    this.headRows = I + 3 + (C - 1);
    this.head = new Float32Array(this.headRows * D);
    const proj = await q.textProject(ins.concat(role));
    this.head.set(proj, 0);
    for (let i = 0; i < C - 1; i++) {
      const t = i < C - 2 ? this.pad : this.bos, o = (I + 3 + i) * D;
      for (let d = 0; d < D; d++) this.head[o + d] = t[d] + cie[i * D + d];
    }
    if (v.mode === "icl") {
      if (!v.refCodes || v.refCodebooks !== tc.groups) fail("голос ICL должен содержать коды образца с " + tc.groups + " кодбуками", "format");
      if (!v.refTextTokens || v.refTextTokens.length < 6) fail("в голосе нет токенов текста образца", "format");
      this.refCodes = v.refCodes; this.refFrames = v.refFrames;
      this.codecLens = 1 + v.refFrames;
      this.codecPart = new Float32Array(this.codecLens * D);
      q.codecRow(tc.bos, this.codecPart.subarray(0, D));
      const tmp = new Float32Array(D);
      for (let f = 0; f < v.refFrames; f++) {
        const dst = (1 + f) * D;
        for (let g = 0; g < tc.groups; g++) {
          const id = v.refCodes[f * v.refCodebooks + g];
          if (g === 0) q.codecRow(id, tmp); else q.row(q.predEmb[g - 1], id, tmp);
          for (let d = 0; d < D; d++) this.codecPart[dst + d] += tmp[d];
        }
      }
      await this.appendText(Array.from(v.refTextTokens.subarray(3, v.refTextTokens.length - 2)));
    } else {
      this.refFrames = 0; this.refCodes = null;
      this.codecLens = 1;
      this.codecPart = new Float32Array(D);
      q.codecRow(tc.bos, this.codecPart);
    }
    return this;
  }

  async appendText(tokens) {
    if (!tokens.length) return;
    const D = this.q.tc.hidden, proj = await this.q.textProject(tokens);
    for (let i = 0; i < tokens.length; i++) this.textStream.push(proj.subarray(i * D, (i + 1) * D));
    this.textN += tokens.length;
  }
  needsText() {
    if (this.textEnded) return false;
    if (!this.prefilled) return this.textN < this.codecLens;
    return this.textN < this.codecLens + this.framesDone;
  }
  _elem(idx) {
    if (idx < this.textN) return this.textStream[idx];
    if (this.textEnded && idx === this.textN) return this.eos;
    return this.pad;
  }

  async _talkerLogits() {
    const q = this.q, D = q.tc.hidden, t0 = performance.now();
    let logits;
    if (!this.prefilled) {
      if (this.textEnded && this.codecLens === 1 && this.textN === 0) fail("пустой текст", "arg");
      const rows = this.headRows + this.codecLens, x = new Float32Array(rows * D);
      x.set(this.head);
      for (let j = 0; j < this.codecLens; j++) {
        const t = this._elem(j), o = (this.headRows + j) * D;
        for (let d = 0; d < D; d++) x[o + d] = t[d] + this.codecPart[j * D + d];
      }
      const cap = rows + this.maxFrames + 16;
      this.ts = q.newTalkerState(cap);
      logits = await q.talkerStep(this.ts, x, rows);
      this.prefilled = true;
      this.stats.prefill_ms += performance.now() - t0; this.stats.prefill_rows = rows;
    } else {
      const t = this._elem(this.codecLens + this.framesDone - 1), x = new Float32Array(D);
      for (let d = 0; d < D; d++) x[d] = this.pending[d] + t[d];
      logits = await q.talkerStep(this.ts, x, 1);
      this.stats.talker_ms += performance.now() - t0;
    }
    return logits;
  }

  async generate() {
    const q = this.q, tc = q.tc, G = tc.groups, D = tc.hidden;
    while (this.buffered < this.chunk) {
      this._abortCheck();
      if (this.needsText()) return;
      if (this.frame >= this.maxFrames) { this.finished = true; return; }
      const logits = await this._talkerLogits();
      const force = this.opts.force;   // тесты: подставить эталонные коды (teacher forcing) и записать логиты шагов
      if (this.opts.trace) this.opts.trace.talker.push(logits.slice());
      if (force && this.frame >= force.length / G) { this.finished = true; return; }
      for (let i = this.suppressFrom; i < this.suppressTo && i < logits.length; i++) if (i !== tc.eos) logits[i] = -Infinity;
      if ((this.noEos || this.frame < this.minEos) && tc.eos >= 0 && tc.eos < logits.length) logits[tc.eos] = -Infinity;
      const c0 = force ? force[this.frame * G] : qSample(logits, this.talker, qUniform(this.k0, this.k1, this.frame, 0), this.history0);
      if (c0 === tc.eos) { this.finished = true; return; }
      this.history0.push(c0);
      this.codes[0] = c0;
      this.flat.push(c0);
      const t0 = performance.now();
      for (let cb = 1; cb < G; cb++) {
        const lg = await q.predictorLogits(this.ts, cb, cb === 1 ? c0 : this.codes[cb - 1]);
        if (this.opts.trace) this.opts.trace.pred.push(lg.slice());
        const c = force ? force[this.frame * G + cb] : qSample(lg, this.sub, qUniform(this.k0, this.k1, this.frame, cb), null);
        this.codes[cb] = c; this.flat.push(c);
      }
      this.stats.predictor_ms += performance.now() - t0;
      // end_frame: сумма эмбеддингов кадра
      const pend = new Float32Array(D), tmp = new Float32Array(D);
      q.codecRow(this.codes[0], tmp);
      for (let d = 0; d < D; d++) pend[d] += tmp[d];
      for (let g = 1; g < G; g++) { q.row(q.predEmb[g - 1], this.codes[g], tmp); for (let d = 0; d < D; d++) pend[d] += tmp[d]; }
      this.pending = pend;
      this.framesDone++; this.frame++; this.buffered++; this.stats.frames++;
    }
  }

  // Прогрев декодера кодека кодами образца (в оригинале они декодируются вместе с речью и обрезаются)
  async _warm() {
    this.warmed = true;
    const q = this.q, G = q.tc.groups, cap = this.refFrames + this.maxFrames + 16;
    this.cs = q.newCodecState(cap);
    if (!this.refFrames) return;
    const ctx = this.ctxFrames >= 0xFFFFFFFF ? this.refFrames : Math.min(this.refFrames, this.ctxFrames || this.refFrames);
    for (let f = this.refFrames - ctx; f < this.refFrames; f += 16) {
      const n = Math.min(16, this.refFrames - f);
      await q.codecForward(this.cs, this.refCodes.subarray(f * G, (f + n) * G), n);
    }
  }
  async _decodeBuffered() {
    if (!this.buffered) return new Float32Array(0);
    if (!this.warmed) await this._warm();
    const t0 = performance.now(), G = this.q.tc.groups, parts = [];
    const codes = Int16Array.from(this.flat);
    let n = 0;
    for (let f = 0; f < this.buffered; f += QCHUNK_MAX_FRAMES) {
      const m = Math.min(QCHUNK_MAX_FRAMES, this.buffered - f);
      const pcm = await this.q.codecForward(this.cs, codes.subarray(f * G, (f + m) * G), m);
      parts.push(pcm); n += pcm.length;
    }
    this.flat = []; this.buffered = 0;
    this.stats.codec_ms += performance.now() - t0;
    if (parts.length === 1) return parts[0];
    const out = new Float32Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  // Один шаг (аналог одного прохода vox_session_pull). → { pcm?, done?, idle? }
  async step() {
    if (this.finished) {
      if (this.flushed) return { done: true };
      const pcm = await this._decodeBuffered();
      this.flushed = true;
      return { pcm, done: true };
    }
    await this.generate();
    if (this.buffered) return { pcm: await this._decodeBuffered() };      // чанк набран или не хватает текста — отдаём, что есть
    if (this.finished) return this.step();
    return { idle: true };
  }

  dispose() {
    if (this.ts) { this.ts.free(); this.ts = null; }
    if (this.cs) { this.q.freeCodecState(this.cs); this.cs = null; }
  }
}

// ---------- модель ----------
class QwenModel {
  constructor(gpu, tm, q) {
    this.gpu = gpu; this.q = q;
    this.manifest = tm.manifest;
    this.tm = tm;
    this.family = "qwen3-tts";
    this.sampleRate = (tm.manifest.audio || {}).sample_rate || 24000;
    this.generation = Object.assign({}, QWEN_DEFAULTS, tm.manifest.generation || {});
    const m = tm.manifest.model || {};
    this.type = m.type || "base";
    this.tok = new QwenTokenizer(new TextDecoder().decode(tm.raw("tokenizer.vocab")), new TextDecoder().decode(tm.raw("tokenizer.merges")), (tm.manifest.tokenizer || {}).specials || []);
    const tcl = q.tc;
    this.speakers = Object.keys(tcl.spk_id).sort();
    this.languages = ["auto"].concat(Object.keys(tcl.lang).sort());
    const small = /0b6|0\.6b/i.test(m.variant || "");
    this.capabilities = {
      named_speakers: this.type === "custom_voice" && this.speakers.length > 0,
      instruct: (this.type === "custom_voice" && !small) || this.type === "voice_design",
      requires_instruct: this.type === "voice_design",
      clone_from_tvoice: this.type === "base", clone_from_audio: this.type === "base" && !!((tm.manifest.qwen3 || {}).speaker_encoder && (tm.manifest.qwen3 || {}).encoder),
      speakers: this.speakers, languages: this.languages, text_in: "token", audio_out: "frames",
    };
    this.info = { name: m.name || "", family: "qwen3-tts", variant: m.variant || "", type: this.type, sampleRate: this.sampleRate, weightBytes: q.wbytes, device: gpu.info, backend: gpu.isCpu ? "cpu" : "webgpu", precision: tm.manifest.precision || {}, frameRate: this.sampleRate / q.cc.samples_per_frame };
  }

  // ---- голоса ----
  voiceFromFile(bytes) {
    const v = QVoice.from(bytes);
    if (v.model && v.model.family && v.model.family !== "qwen3-tts") fail("голос создан для другого семейства: " + v.model.family, "arg");
    if (v.mode !== "speaker" && v.speakerEmb.length !== this.q.tc.hidden) fail("голос создан для модели другого размера (вектор диктора " + v.speakerEmb.length + ", нужен " + this.q.tc.hidden + ")", "arg");
    return v;
  }
  // Готовый диктор (CustomVoice) и/или описание голоса (VoiceDesign, CustomVoice 1.7B)
  speakerVoice(speaker, instruct, opts = {}) {
    const c = this.capabilities;
    let spk = String(speaker || ""), ins = String(instruct || "");
    if (!spk && !ins) fail("укажите диктора, описание голоса или и то и другое", "arg");
    if (spk) {
      if (!c.named_speakers) fail("в этой модели нет встроенных дикторов (нужна модель CustomVoice или клонирование голоса из записи)", "arg");
      spk = spk.toLowerCase();
      if (!c.speakers.includes(spk)) fail("неизвестный диктор '" + speaker + "'; в модели есть: " + c.speakers.join(", "), "arg");
    }
    if (ins && !c.instruct) fail("эта модель не принимает описание голоса/стиля (умеют VoiceDesign и CustomVoice 1.7B)", "arg");
    if (c.requires_instruct) {
      if (!ins) fail("эта модель (VoiceDesign) требует описание голоса", "arg");
      if (spk) fail("в VoiceDesign нет именованных дикторов: опишите голос текстом", "arg");
    }
    if (!c.named_speakers && !c.requires_instruct) fail("эта модель не создаёт голоса без записи", "arg");
    return new QVoice({ mode: "speaker", name: spk || "designed", speaker: spk, instruct: ins, language: opts.language || "auto" });
  }
  // Голос из записи (Base-модели): вектор диктора (ECAPA-TDNN) + коды образца (Mimi), всё на CPU маленькими порциями.
  // audio: Float32Array (+ opts.sampleRate), AudioBuffer, Blob/File, ArrayBuffer, URL. refText — что сказано в записи (нужен для icl).
  // opts: { xvectorOnly, language, name, onProgress(доля), signal }
  async cloneVoice(audio, refText, opts = {}) {
    if (!this.capabilities.clone_from_audio) fail("эта модель не создаёт голос по записи (нужна Base-модель: " + (this.type === "base" ? "в загруженном файле нет энкодеров голоса, соберите его из чекпойнта HF" : "тип " + this.type + ")"), "unsupported");
    const xv = !!opts.xvectorOnly;
    if (!xv && (typeof refText !== "string" || !refText.trim())) fail("нужен текст образца (refText): что произносится в записи (или xvectorOnly: true)", "arg");
    let pcm, sr;
    if (audio instanceof Float32Array) { pcm = audio; sr = opts.sampleRate; if (!sr) fail("для Float32Array укажите opts.sampleRate", "arg"); }
    else ({ pcm, sampleRate: sr } = await decodeAudio(audio));
    let a = resampleSinc(pcm, sr, 24000);
    if (a.length < 24000) fail("образец слишком короткий: нужно хотя бы пара секунд речи", "arg");
    const maxS = opts.maxSeconds || 30;
    if (a.length > maxS * 24000) a = a.slice(0, maxS * 24000);   // длинный образец только замедляет, а голос от него не лучше
    const prog = opts.onProgress || (() => {});
    const abort = () => { if (opts.signal && opts.signal.aborted) fail("создание голоса отменено", "aborted"); };
    const y = new Yielder(24, abort);
    if (!this._venc) this._venc = await QwenVoiceEncoder.create(this.tm, y);
    this._venc.y = y;
    const enc = this._venc, t0 = performance.now();
    prog(0.02);
    const emb = await enc.speakerEmbedding(a);
    if (emb.length !== this.q.tc.hidden) fail("вектор диктора не совпал с размером talker", "format");
    prog(xv ? 1 : 0.25);
    const d = { mode: xv ? "xvector_only" : "icl", name: opts.name || "", refText: xv ? "" : refText.trim(), language: opts.language || "auto", speakerEmb: emb,
      model: { family: "qwen3-tts", variant: (this.manifest.model || {}).variant || "", weights_sha256: (this.manifest.model || {}).weights_sha256 || "" } };
    if (!xv) {
      const r = await enc.encode(a, (f) => prog(0.25 + 0.75 * f));
      if (r.codebooks !== this.q.tc.groups) fail("энкодер речи дал " + r.codebooks + " кодбуков, а talker ждёт " + this.q.tc.groups, "format");
      d.refCodes = r.codes; d.refFrames = r.frames; d.refCodebooks = r.codebooks;
      d.refTextTokens = Int32Array.from(this.tok.encode("<|im_start|>assistant\n" + refText.trim() + "<|im_end|>\n"));
    }
    const v = new QVoice(d);
    v.seconds = a.length / 24000; v.createMs = performance.now() - t0;
    return v;
  }

  _check(voice) {
    if (!(voice instanceof QVoice)) fail("передайте голос: opts.voice = model.speakerVoice(...) или model.voiceFromFile(bytes)", "arg");
    return voice;
  }

  // Весь текст сразу → поток звука (кусками по chunk_frames кадров)
  async *stream(text, opts = {}) {
    if (typeof text !== "string" || !text.trim()) fail("пустой текст", "arg");
    const voice = this._check(opts.voice), t0 = performance.now();
    const s = new QwenSession(this, voice, opts);
    try {
      await s.init();
      await s.appendText(this.tok.encode(text.normalize("NFC")));
      s.textEnded = true;
      let produced = 0;
      for (;;) {
        const r = await s.step();
        if (r.pcm && r.pcm.length) { produced += r.pcm.length; yield r.pcm; }
        if (r.done) break;
      }
      this.lastRun = { audioSeconds: produced / this.sampleRate, wallMs: performance.now() - t0, frames: s.stats.frames, timing: Object.assign({}, s.stats) };
    } finally { s.dispose(); }
  }
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
    const cfg = { sample_rate: this.sampleRate, n_fft: 1024, hop: 256, win: 1024, n_mels: 100 };
    if (!this._melFb) this._melFb = melFilterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels);
    const r = melLogSpectrogram(cfg, this._melFb, pcm);
    return { mel: r.mel, frames: r.frames, bands: cfg.n_mels };
  }
  dispose() { if (this._venc) this._venc.release(); this.q.dispose(); try { this.gpu.device.destroy(); } catch (e) { /* ок */ } }
}
