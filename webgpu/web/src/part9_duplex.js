// ============================================================================
// Двусторонний стриминг в браузере: текст подаётся на лету, звук приходит на лету. Та же логика, что в ядре vox
// (vox.h ABI 0.4): политика коммита auto | sentence | manual, idle_ms, flush, правка ещё не озвученного хвоста.
// Дуплекс — общая оболочка; модель подключается адаптером (для F5-TTS — F5Stream, озвучивает куски целиком).
// ============================================================================
const _utf8 = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
const utf8len = (s) => _utf8.encode(s).length;

// Конец последнего законченного предложения (индекс в строке JS), 0 — нет. Знак конца законченный, если за ним пробел/перевод строки
// или он полноширинный (。！？…); final — конец текста. "3.5", "e.g." и "..." посередине не рвутся. Порт sentence_cut из ядра.
function sentenceCut(t, final) {
  let best = 0;
  const isWs = (c) => c === " " || c === "\t" || c === "\n" || c === "\r";
  for (let i = 0; i < t.length;) {
    const c = t[i];
    const full = c === "。" || c === "！" || c === "？" || c === "…";
    const term = full || c === "." || c === "!" || c === "?" || c === "\n";
    const len = 1;   // все знаки конца — одиночные кодовые единицы UTF-16
    if (term) {
      let j = i + len;
      const wsAfter = j < t.length && isWs(t[j]);
      if (full || wsAfter || final || c === "\n") {
        while (j < t.length && isWs(t[j])) j++;
        const glued = !full && j === i + len && j < t.length && (t[j] === "." || t[j] === "!" || t[j] === "?");
        if (!glued) best = j;
      }
    }
    i += len;
  }
  return best;
}

// Адаптер F5-TTS (порт direct_text / direct_seal / direct_step из backend_f5.cpp): куски озвучиваются целиком.
class F5Stream {
  constructor(model, voice, s, opts) {
    this.model = model; this.voice = voice; this.s = s; this.opts = opts;
    this.P = model._plan(voice, s);
    this.sp = new Splicer(this.P.cfMax);
    this.seg = ""; this.sealed = false; this.nextChunk = 0; this.counter = 0; this.flushedEnd = false;
    this.caps = { text_in: "chunk", audio_out: "chunks", exact: true };
    this.direct = true;
    model.f5.timing = { dit_ms: 0, vocos_ms: 0, text_ms: 0 };
  }
  commit(piece, seal) { this.seg += piece; if (seal) this.sealed = true; }
  seal() { this.sealed = true; }
  safeCut(pending) { return pending.length; }
  async step(ended) {
    const chunks = chunkText(this.seg, this.P.maxChars);
    const avail = (ended || this.sealed) ? chunks.length : Math.max(0, chunks.length - 1);
    if (this.nextChunk < avail) {
      const wave = await this.model._chunkWave(this.voice, this.s, this.P, chunks[this.nextChunk], this.counter, this.opts);
      const piece = this.sp.push(wave, this.counter === 0);
      this.nextChunk++; this.counter++;
      return { pcm: piece };
    }
    if (this.sealed && !ended) {
      const tail = this.sp.release(true);
      this.seg = ""; this.nextChunk = 0; this.sealed = false;
      return { pcm: tail };
    }
    if (!ended) return { idle: true };
    if (!this.flushedEnd) { this.flushedEnd = true; return { pcm: this.sp.release(false), done: true }; }
    return { done: true };
  }
}

// Адаптер авторегрессионных моделей (Qwen3-TTS): текст подаётся токенами, звук идёт покадрово. Высказывание = одна сессия
// генерации; пауза/правка закрывает текущее высказывание (оно дочитывается), следующий текст открывает новое (seed + номер).
class QwenStream {
  constructor(model, voice, opts) {
    this.model = model; this.voice = voice; this.opts = opts;
    this.caps = { text_in: "token", audio_out: "frames", exact: true };
    this.direct = false;
    this.sessions = []; this.open = null; this.count = 0;
    this.baseSeed = Math.trunc(opts.seed != null ? opts.seed : (voice.sampling && voice.sampling.seed) || 1) || 1;
  }
  _new() {
    const o = Object.assign({}, this.opts, { seed: this.baseSeed + this.count++ });
    const e = { sess: new QwenSession(this.model, this.voice, o), queue: [], closing: false, ready: false };
    this.sessions.push(e);
    return e;
  }
  commit(piece) {
    const toks = this.model.tok.encode(piece);
    if (!toks.length) return;
    if (!this.open) this.open = this._new();
    for (const t of toks) this.open.queue.push(t);
  }
  seal() { /* у AR-модели нечего «запечатывать»: хвост слова уже закоммичен, фраза закончится с паузой/end */ }
  safeCut(pending) { return this.model.tok.safeCut(pending); }
  endUtterance() { if (this.open) { this.open.closing = true; this.open = null; } }
  gap() { this.endUtterance(); }
  async step(ended) {
    if (ended) this.endUtterance();
    const e = this.sessions[0];
    if (!e) return ended ? { done: true } : { idle: true };
    const s = e.sess;
    if (!e.ready) { await s.init(); e.ready = true; }
    if (e.queue.length) { const t = e.queue; e.queue = []; await s.appendText(t); }
    if (e.closing) s.textEnded = true;
    const r = await s.step();
    if (r.done) { s.dispose(); this.sessions.shift(); return { pcm: r.pcm }; }
    return r;
  }
  dispose() { for (const e of this.sessions) e.sess.dispose(); this.sessions = []; }
}

class Duplex {
  constructor(model, opts = {}) {
    const voice = opts.voice;
    const commit = opts.commit || "auto";
    if (!["auto", "sentence", "manual"].includes(commit)) fail("commit: auto | sentence | manual", "arg");
    const onConflict = opts.on_conflict || "keep";
    if (!["keep", "raise", "respeak"].includes(onConflict)) fail("on_conflict: keep | raise | respeak", "arg");
    this.model = model; this.voice = voice;
    this.commit = commit; this.idleMs = Math.max(0, Number(opts.idle_ms) || 0); this.gapMs = Math.max(0, Number(opts.utterance_gap_ms) || 0);
    this.onConflict = onConflict;
    this.sampleRate = model.sampleRate;
    this.ac = new AbortController();
    if (opts.signal) opts.signal.addEventListener("abort", () => this.cancel());
    this.be = model._duplexBackend(voice, opts, this.ac.signal);
    this.caps = this.be.caps;
    this.pending = ""; this.committedText = "";
    this.ended = false; this.cancelled = false; this.done = false; this.error = null;
    this.flushes = 0; this.idleFlushes = 0; this.idleArmed = false; this.lastFeed = 0; this.lastText = 0; this.gapFired = true;
    this.utterances = 1;
    this.q = []; this._aw = null; this._wake = null;
    this.stats = { firstAudioMs: null, audioSamples: 0 };
    this._tFirstWrite = null;
    this._pump();
  }

  // ---------- вход ----------
  _commitPending(force) {
    const t = this.pending;
    if (!t) return 0;
    let cut = 0;
    if (force) cut = t.length;
    else if (this.commit === "sentence") cut = sentenceCut(t, false);
    else if (this.commit === "auto") cut = this.be.safeCut(t);
    if (!cut) return 0;
    const piece = t.slice(0, cut);
    this.be.commit(piece, this.be.direct && (force || this.commit !== "auto"));
    this.pending = t.slice(cut);
    this.committedText += piece;
    return cut;
  }
  _poke() { this._ver = (this._ver || 0) + 1; if (this._wake) { const w = this._wake; this._wake = null; w(); } }
  write(text) {
    if (this.ended) fail("текст уже закончен (end())", "state");
    if (this.cancelled) fail("сессия отменена", "state");
    if (!text) return this;
    if (this._tFirstWrite == null) this._tFirstWrite = performance.now();
    this.pending += String(text).normalize("NFC");
    this.lastFeed = this.lastText = performance.now(); this.idleArmed = true; this.gapFired = false;
    this._commitPending(false);
    this._poke();
    return this;
  }
  flush() {
    if (this.ended || this.cancelled) return this;
    if (this._commitPending(true)) this.flushes++;
    else if (this.be.direct) this.be.seal();
    this.idleArmed = false;
    this._poke();
    return this;
  }
  // Режим набора: передаётся ВЕСЬ текст из поля ввода. → { ok, committed, pending, spoken? } (committed — в байтах UTF-8)
  setText(full) {
    if (this.ended) fail("текст уже закончен (end())", "state");
    full = String(full).normalize("NFC");
    if (this._tFirstWrite == null && full) this._tFirstWrite = performance.now();
    const logical = this.committedText + this.pending;
    const cl = this.committedText.length;
    let res;
    if (full.slice(0, cl) === this.committedText) {
      this.pending = full.slice(cl);
      this.lastFeed = this.lastText = performance.now(); this.idleArmed = !!this.pending; this.gapFired = false;
      this._commitPending(false);
      res = { ok: true, committed: utf8len(this.committedText), pending: this.pending };
    } else {
      res = { ok: false, committed: utf8len(this.committedText), spoken: this.committedText, pending: this.pending };
      if (this.onConflict === "raise") fail("правка внутри уже озвученного текста (озвучено: " + JSON.stringify(this.committedText) + ")", "conflict");
      if (this.onConflict === "respeak") {
        let i = 0;
        while (i < Math.min(full.length, cl) && full[i] === logical[i]) i++;
        const j = full.lastIndexOf(" ", i - 1) + 1;     // с начала изменённого слова
        this.flush();
        if (this.be.endUtterance) this.be.endUtterance();
        this.committedText = "";
        this.pending = full.slice(j);
        this._commitPending(false);
        this.utterances++;
        res = { ok: true, committed: utf8len(this.committedText), pending: this.pending, respeakFrom: j };
      }
    }
    this._poke();
    return res;
  }
  end() {
    if (this.ended) return this;
    this._commitPending(true);
    this.ended = true;
    this._poke();
    return this;
  }
  cancel() {
    this.cancelled = true; this.ended = true;
    try { this.ac.abort(); } catch (e) { /* ок */ }
    this._poke();
  }
  info() {
    return { committed_bytes: utf8len(this.committedText), pending: this.pending, pending_bytes: utf8len(this.pending), text_ended: this.ended,
      finished: this.done, commit: this.commit, idle_ms: this.idleMs, flushes: this.flushes, idle_flushes: this.idleFlushes, utterances: this.utterances };
  }

  // ---------- рабочий цикл ----------
  _sleep(ms) {
    return new Promise((res) => {
      const t = ms > 0 ? setTimeout(() => { this._wake = null; res(); }, ms) : null;
      this._wake = () => { if (t) clearTimeout(t); res(); };
    });
  }
  _emit(pcm) {
    if (!pcm || !pcm.length) return;
    if (this.stats.firstAudioMs == null && this._tFirstWrite != null) this.stats.firstAudioMs = performance.now() - this._tFirstWrite;
    this.stats.audioSamples += pcm.length;
    this.q.push(pcm);
    if (this._aw) { const w = this._aw; this._aw = null; w(); }
  }
  async _pump() {
    try {
      for (;;) {
        if (this.cancelled) break;
        // простой: автокоммит накопленного
        if (this.idleMs && this.idleArmed && !this.ended && this.pending && performance.now() - this.lastFeed >= this.idleMs) {
          this.idleArmed = false;
          if (this._commitPending(true)) { this.flushes++; this.idleFlushes++; }
        }
        // пауза между высказываниями: для F5 это то же, что flush (хвост отдаётся, следующий текст — с нуля)
        if (this.gapMs && !this.gapFired && !this.ended && performance.now() - this.lastText >= this.gapMs && (this.committedText || this.pending)) {
          this.gapFired = true;
          if (this.be.gap) { if (this._commitPending(true)) this.flushes++; this.be.gap(); } else this.flush();
        }
        const ver = this._ver || 0;
        const r = await this.be.step(this.ended);
        if (r.pcm) this._emit(r.pcm);
        if (r.done) break;
        if (r.idle && (this._ver || 0) === ver) {   // не засыпаем, если пока шёл шаг, пришёл новый текст (иначе потеряется «толчок»)
          let wait = 0;
          if (this.idleMs && this.idleArmed && this.pending) wait = Math.max(1, this.idleMs - (performance.now() - this.lastFeed));
          if (this.gapMs && !this.gapFired) wait = wait ? Math.min(wait, this.gapMs) : this.gapMs;
          await this._sleep(wait);
        }
      }
    } catch (e) {
      if (!(e instanceof VoxError && e.code === "aborted")) this.error = e;
    } finally {
      if (this.be.dispose) this.be.dispose();
      this.done = true;
      if (this._aw) { const w = this._aw; this._aw = null; w(); }
    }
  }

  // ---------- выход ----------
  // Асинхронный итератор по кускам звука до конца (после end() или cancel()).
  async *audio() {
    for (;;) {
      if (this.q.length) { yield this.q.shift(); continue; }
      if (this.done) { if (this.error) throw this.error; return; }
      await new Promise((res) => { this._aw = res; });
    }
  }
  // Один кусок или null по таймауту/концу (проверяйте .done).
  async read(timeoutMs) {
    if (this.q.length) return this.q.shift();
    if (this.done) { if (this.error) throw this.error; return null; }
    await Promise.race([new Promise((res) => { this._aw = res; }), new Promise((res) => setTimeout(res, timeoutMs == null ? 1e9 : timeoutMs))]);
    this._aw = null;
    return this.q.length ? this.q.shift() : null;
  }
  // Закрывает текст, ждёт, отдаёт всю запись одним массивом.
  async final() {
    this.end();
    const parts = [];
    let n = 0;
    for await (const c of this.audio()) { parts.push(c); n += c.length; }
    const pcm = new Float32Array(n);
    let o = 0;
    for (const p of parts) { pcm.set(p, o); o += p.length; }
    return { pcm, sampleRate: this.sampleRate, seconds: n / this.sampleRate, stats: Object.assign({}, this.stats) };
  }
}

VoxModel.prototype.duplex = function (opts) { return new Duplex(this, opts); };
QwenModel.prototype.duplex = function (opts) { return new Duplex(this, opts); };
VoxModel.prototype._duplexBackend = function (voice, opts, signal) {
  if (!(voice instanceof Voice)) fail("передайте голос: model.duplex({ voice })", "arg");
  if (voice.melDim !== this.f5.cfg.mel_dim) fail("голос создан для другой модели", "arg");
  return new F5Stream(this, voice, this._opts(opts), { signal, onStep: opts.onStep, noise: opts.noise });
};
QwenModel.prototype._duplexBackend = function (voice, opts, signal) {
  this._check(voice);
  return new QwenStream(this, voice, Object.assign({}, opts, { signal }));
};
Vox.Duplex = Duplex;
Vox._internal.sentenceCut = sentenceCut;
Vox._internal.Splicer = Splicer;
