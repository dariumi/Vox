// ============================================================================
// Qwen3-TTS: загрузка чекпойнта HF (safetensors) прямо в браузере. Те же преобразования, что в python/vox/build.py
// (свёртки под im2col, кодбук = embedding_sum / cluster_usage, q8, словарь/merges токенизатора), но без сборки целого .tmod
// в памяти: источник отдаёт тензоры по требованию (HTTP Range по заголовку safetensors), каждый сразу уходит в GPU и забывается.
// Интерфейс — как у Tmod (has / info / raw / f32 / rawAsync / release) плюс prepare / warm.
// ============================================================================
const QWEN_TALKER_DEFAULTS = {
  vocab_size: 3072, hidden_size: 1024, intermediate_size: 2048, num_hidden_layers: 20, num_attention_heads: 16, num_key_value_heads: 2,
  rms_norm_eps: 1e-6, rope_theta: 10000, num_code_groups: 32, text_hidden_size: 2048, text_vocab_size: 151936, codec_eos_token_id: 4198,
  codec_think_id: 4202, codec_nothink_id: 4203, codec_think_bos_id: 4204, codec_think_eos_id: 4205, codec_pad_id: 4196, codec_bos_id: 4197,
};
const QWEN_PRED_DEFAULTS = {
  vocab_size: 2048, hidden_size: 1024, intermediate_size: 3072, num_hidden_layers: 5, num_attention_heads: 16, num_key_value_heads: 8,
  head_dim: 128, rms_norm_eps: 1e-6, rope_theta: 10000, num_code_groups: 32,
};
const QWEN_CODEC_DEFAULTS = {
  codebook_size: 2048, hidden_size: 1024, latent_dim: 1024, rope_theta: 10000, num_attention_heads: 16, num_key_value_heads: 16,
  sliding_window: 72, intermediate_size: 3072, layer_scale_initial_scale: 0.01, rms_norm_eps: 1e-5, num_hidden_layers: 8,
  num_quantizers: 16, upsample_rates: [8, 5, 4, 3], upsampling_ratios: [2, 2], decoder_dim: 1536,
};
const QWEN_IDS_DEFAULTS = { im_start_token_id: 151644, im_end_token_id: 151645, tts_pad_token_id: 151671, tts_bos_token_id: 151672, tts_eos_token_id: 151673 };
const QWEN_SPK_DEFAULTS = { mel_dim: 128, enc_dim: 1024, enc_channels: [512, 512, 512, 512, 1536], enc_kernel_sizes: [5, 3, 3, 3, 1], enc_dilations: [1, 2, 3, 4, 1],
  enc_attention_channels: 128, enc_res2net_scale: 8, enc_se_channels: 128, sample_rate: 24000 };
const QWEN_MIMI_KEYS = ["sampling_rate", "audio_channels", "hidden_size", "num_filters", "num_residual_layers", "upsampling_ratios", "kernel_size", "last_kernel_size",
  "residual_kernel_size", "dilation_growth_rate", "use_causal_conv", "pad_mode", "compress", "codebook_size", "codebook_dim", "num_quantizers", "use_conv_shortcut",
  "vector_quantization_hidden_dimension", "num_semantic_quantizers", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
  "hidden_act", "norm_eps", "rope_theta", "frame_rate"];
const QWEN_MIMI_DEFAULTS = { sampling_rate: 24000, audio_channels: 1, hidden_size: 512, num_filters: 64, num_residual_layers: 1, upsampling_ratios: [8, 6, 5, 4], kernel_size: 7,
  last_kernel_size: 3, residual_kernel_size: 3, dilation_growth_rate: 2, use_causal_conv: true, pad_mode: "constant", compress: 2, codebook_size: 2048, codebook_dim: 256,
  num_quantizers: 32, use_conv_shortcut: false, vector_quantization_hidden_dimension: 256, num_semantic_quantizers: 1, num_hidden_layers: 8, intermediate_size: 2048,
  num_attention_heads: 8, num_key_value_heads: 8, head_dim: null, hidden_act: "gelu", norm_eps: 1e-5, rope_theta: 10000.0, frame_rate: 12.5 };
const QWEN_LINEAR_Q8 = new RegExp("(\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)\\.weight$)|(^talker\\.codec_head\\.weight$)|" +
  "(^talker\\.code_predictor\\.lm_head\\.\\d+\\.weight$)|(^talker\\.code_predictor\\.small_to_mtp_projection\\.weight$)|(^talker\\.text_projection\\.linear_fc[12]\\.weight$)");
const qMerged = (defaults, given) => { const o = Object.assign({}, defaults); for (const k of Object.keys(given || {})) if (given[k] != null) o[k] = given[k]; return o; };

// ---------- чтение диапазонов ----------
class Sem {
  constructor(n) { this.n = n; this.q = []; }
  async run(fn) {
    if (this.n <= 0) await new Promise((r) => this.q.push(r)); else this.n--;
    try { return await fn(); } finally { const nx = this.q.shift(); if (nx) nx(); else this.n++; }
  }
}

// Файл, из которого можно читать куски: байты в памяти, Blob/File или URL с поддержкой Range
class RangeFile {
  constructor(label, size, readFn) { this.label = label; this.size = size; this._read = readFn; }
  async read(a, b) { if (b <= a) return new Uint8Array(0); return this._read(a, b); }
  static fromBytes(label, bytes) { return new RangeFile(label, bytes.length, async (a, b) => bytes.subarray(a, b)); }
  static fromBlob(label, blob) { return new RangeFile(label, blob.size, async (a, b) => new Uint8Array(await blob.slice(a, b).arrayBuffer())); }
  // URL с поддержкой Range. Читает крупными блоками (по умолчанию 8 МиБ, чтобы не платить за запрос на каждый из сотен тензоров:
  // у HF каждый запрос — это ещё и редирект на CDN и предзапрос CORS) и запоминает адрес CDN после первого редиректа.
  static async fromUrl(label, url, opts, tally) {
    opts = opts || {};
    const net = opts._net || (opts._net = { requests: 0, bytes: 0, ms: 0, redirects: 0, cached: 0 });
    const BLOCK = Math.max(1 << 20, (opts.blockMiB || 8) * 1048576), KEEP = Math.max(4, opts.keepBlocks || 12);
    let target = url, whole = null, total = 0, etag = "";
    // Кэш блоков в Cache API: повторная загрузка той же модели не качает гигабайты заново. Ключ включает размер файла и ETag (без них кэш не используется).
    let store = null;
    const cacheKey = (i) => url + (url.includes("?") ? "&" : "?") + "voxblock=" + i + "&bs=" + BLOCK + "&size=" + total + "&etag=" + encodeURIComponent(etag);
    const openStore = async () => { if (store === null) { try { store = opts.cache === false || typeof caches === "undefined" || !total || !etag ? false : await caches.open("vox-blocks-v1"); } catch (e) { store = false; } } return store; };
    const headers = (u, extra) => Object.assign({}, qwenAuth(u, opts), extra || {});
    const fetchRange = async (a, b) => {
      for (let attempt = 0; ; attempt++) {
        const t0 = performance.now();
        let r;
        try { r = await fetch(target, { headers: headers(target, { Range: "bytes=" + a + "-" + (b - 1) }) }); }
        catch (e) {
          if (target !== url) { target = url; continue; }
          if (attempt < 2) { await new Promise((q) => setTimeout(q, 400 * (attempt + 1))); continue; }
          throw e;
        }
        if (target === url && r.redirected && r.url && r.url !== url) { target = r.url; net.redirects++; }   // подписанный адрес CDN живёт около часа
        if ((r.status === 403 || r.status === 400) && target !== url) { target = url; continue; }
        if (r.status === 401 || r.status === 403) fail("Hugging Face: доступ закрыт (нужен токен: opts.token): " + label, "network");
        if (!r.ok) {
          if (attempt < 2 && r.status >= 500) { await new Promise((q) => setTimeout(q, 400 * (attempt + 1))); continue; }
          fail("не удалось скачать " + label + ": HTTP " + r.status, "network");
        }
        const buf = new Uint8Array(await r.arrayBuffer());
        net.requests++; net.bytes += buf.length; net.ms += performance.now() - t0;
        if (!total) { const m = /\/(\d+)\s*$/.exec(r.headers.get("content-range") || ""); if (m) total = Number(m[1]); etag = r.headers.get("etag") || ""; }
        return { buf, partial: r.status === 206 };
      }
    };
    const blocks = new Map();   // индекс блока → Promise<Uint8Array>
    const block = (i) => {
      let p = blocks.get(i);
      if (p) { blocks.delete(i); blocks.set(i, p); return p; }   // обновить порядок (LRU)
      p = (async () => {
        const st = await openStore();
        if (st) {
          try { const hit = await st.match(cacheKey(i)); if (hit) { net.cached++; return new Uint8Array(await hit.arrayBuffer()); } } catch (e) { /* ок */ }
        }
        const { buf, partial } = await fetchRange(i * BLOCK, (i + 1) * BLOCK);
        if (!partial) { whole = buf; return buf.subarray(i * BLOCK, (i + 1) * BLOCK); }   // сервер не понимает Range
        if (st && buf.length) st.put(cacheKey(i), new Response(buf.slice())).catch(() => {});   // в фоне; при нехватке места просто не сохранится
        return buf;
      })();
      blocks.set(i, p);
      p.catch(() => blocks.delete(i));
      while (blocks.size > KEEP) blocks.delete(blocks.keys().next().value);
      return p;
    };
    const get = async (a, b) => {
      if (whole) return whole.subarray(a, b);
      if (b - a >= BLOCK) {   // крупный тензор (эмбеддинги): блоки пачками по четыре, в общий список недавних они не попадают
        const out = new Uint8Array(b - a), j0 = Math.floor(a / BLOCK), j1 = Math.floor((b - 1) / BLOCK);
        for (let j = j0; j <= j1; j += 4) {
          const ids = []; for (let q = j; q <= Math.min(j1, j + 3); q++) ids.push(q);
          const got = await Promise.all(ids.map((q) => block(q)));
          ids.forEach((q, k) => {
            const bl = got[k], lo = Math.max(a, q * BLOCK), hi = Math.min(b, q * BLOCK + bl.length);
            out.set(bl.subarray(lo - q * BLOCK, hi - q * BLOCK), lo - a);
            blocks.delete(q);
          });
          if (whole) return whole.subarray(a, b);
        }
        return out;
      }
      const i0 = Math.floor(a / BLOCK), i1 = Math.floor((b - 1) / BLOCK);
      if (i0 === i1) { const bl = await block(i0); return whole ? whole.subarray(a, b) : bl.subarray(a - i0 * BLOCK, b - i0 * BLOCK); }
      const out = new Uint8Array(b - a);
      for (let i = i0; i <= i1; i++) {
        const bl = await block(i), lo = Math.max(a, i * BLOCK), hi = Math.min(b, i * BLOCK + bl.length);
        out.set(bl.subarray(lo - i * BLOCK, hi - i * BLOCK), lo - a);
      }
      return out;
    };
    const head = (await fetchRange(0, 8)).buf;
    const n = Number(new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true));
    const f = new RangeFile(label, 0, get);
    f.headerLen = n; f.net = net;
    return f;
  }
}
function qwenAuth(url, opts) {
  if (!opts || !opts.token) return {};
  const host = (opts.hfHost || HF_HOST).replace(/\/+$/, "");
  return url.startsWith(host + "/") || /^https:\/\/([\w-]+\.)?huggingface\.co\//.test(url) ? { Authorization: "Bearer " + opts.token } : {};
}

// заголовок safetensors из RangeFile → Map имя → { dtype, shape, a, b, file }
async function readStHeader(file) {
  const h8 = await file.read(0, 8);
  const n = Number(new DataView(h8.buffer, h8.byteOffset, 8).getBigUint64(0, true));
  if (n <= 0 || n > 100 * 1024 * 1024) fail("safetensors: странный размер заголовка в " + file.label + " (это не safetensors?)", "format");
  const hdr = JSON.parse(new TextDecoder().decode(await file.read(8, 8 + n)));
  const base = 8 + n, out = new Map();
  for (const k of Object.keys(hdr)) {
    if (k === "__metadata__") continue;
    const t = hdr[k];
    out.set(k, { dtype: t.dtype, shape: t.shape.slice(), a: base + t.data_offsets[0], b: base + t.data_offsets[1], file });
  }
  return out;
}

// ---------- поэлементное преобразование типов без промежуточного f32-массива ----------
const U16 = (raw) => {
  if ((raw.byteOffset & 1) === 0) return new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
  return new Uint16Array(raw.slice().buffer, 0, raw.byteLength >> 1);
};
function convertRaw(raw, src, dst) {
  if (src === dst) return raw;
  const numel = raw.byteLength / ST_SIZE[src];
  if (!ST_SIZE[src] || !ST_SIZE[dst]) fail("неподдерживаемый тип: " + src + " → " + dst, "format");
  if (dst === "F32") {
    const out = new Float32Array(numel);
    if (src === "BF16") { const u = U16(raw), w = new Uint32Array(out.buffer); for (let i = 0; i < numel; i++) w[i] = u[i] << 16; }
    else { const f = rawToF32(raw, src, numel); out.set(f); }
    return new Uint8Array(out.buffer);
  }
  const out = new Uint16Array(numel);
  if (src === "F32") {
    const f = (raw.byteOffset & 3) === 0 ? new Float32Array(raw.buffer, raw.byteOffset, numel) : new Float32Array(raw.slice().buffer, 0, numel);
    if (dst === "F16") for (let i = 0; i < numel; i++) out[i] = f32ToF16Bits(f[i]); else for (let i = 0; i < numel; i++) out[i] = f32ToBf16Bits(f[i]);
  } else {
    const u = U16(raw);
    if (src === "BF16") { for (let i = 0; i < numel; i++) out[i] = f32ToF16Bits(bf16ToF32(u[i])); }   // BF16 → F16
    else { for (let i = 0; i < numel; i++) out[i] = f32ToBf16Bits(f16ToF32(u[i])); }                   // F16 → BF16
  }
  return new Uint8Array(out.buffer);
}
// f32-массив → байты нужного типа (F32/F16/BF16)
function f32ToRaw(f, dst) {
  if (dst === "F32") return new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  const out = new Uint16Array(f.length);
  if (dst === "F16") for (let i = 0; i < f.length; i++) out[i] = f32ToF16Bits(f[i]); else for (let i = 0; i < f.length; i++) out[i] = f32ToBf16Bits(f[i]);
  return new Uint8Array(out.buffer);
}

// ---------- источник ----------
class QwenHfSource {
  constructor() {
    this.manifest = null; this.table = new Map(); this.pre = new Map();
    this.prom = new Map(); this.done = new Map(); this.sem = new Sem(6);
    this.fetched = 0; this.total = 0; this.onProgress = null;
    this._tally = (n) => { this.fetched += n; if (this.onProgress) this.onProgress(Math.min(1, this.fetched / Math.max(1, this.total)), this.fetched, this.total, "weights"); };
  }
  has(name) { return this.table.has(name); }
  info(name) {
    const e = this.table.get(name);
    if (!e) fail("в чекпойнте нет тензора: " + name, "format");
    return e;
  }
  raw(name) {
    const r = this.pre.get(name);
    if (!r) fail("тензор ещё не загружен (используйте rawAsync): " + name, "internal");
    return r;
  }
  f32(name) {
    const r = this.done.get(name);
    if (!r) fail("тензор ещё не подготовлен: " + name, "internal");
    return new Float32Array(r.buffer, r.byteOffset, r.byteLength >> 2);
  }
  _get(name) {
    if (this.pre.has(name)) return Promise.resolve(this.pre.get(name));
    let p = this.prom.get(name);
    if (!p) {
      const e = this.info(name);
      p = this.sem.run(async () => { const r = await e.produce(this); await _macro(); return r; }).then((r) => { this.done.set(name, r); return r; });   // после каждого тензора страница получает слово (ввод, отрисовка)
      this.prom.set(name, p);
      p.catch(() => {});
    }
    return p;
  }
  async prepare(name) { await this._get(name); }
  warm(names) { for (const n of names) if (this.table.has(n)) this._get(n); }
  async rawAsync(name) {
    if (this.pre.has(name)) return this.pre.get(name);
    const e = this.info(name);
    // у q8-пары главный тензор считает и масштаб; ждём того, кто его произведёт
    const r = await this._get(e.owner || name);
    const out = e.owner ? this.done.get(name) : r;
    if (!out) fail("внутренняя ошибка: нет данных тензора " + name, "internal");
    return out;
  }
  release(name) { this.prom.delete(name); this.done.delete(name); const e = this.table.get(name); if (e && e.companion) { this.prom.delete(e.companion); this.done.delete(e.companion); } }
}

// Собирает QwenHfSource. fs: { has(path), text(path), file(path) → Promise<RangeFile> }
async function openQwenSource(fs, repoName, opts = {}) {
  const status = (s) => { if (opts.onStatus) opts.onStatus(s); };
  const dtype = (opts.dtype || "bf16").toUpperCase(), quant = opts.quant || "", codecDtype = (opts.codecDtype || "f32").toUpperCase();
  if (!["F32", "F16", "BF16"].includes(dtype)) fail("dtype: f32, f16 или bf16", "arg");
  if (!["F32", "F16", "BF16"].includes(codecDtype)) fail("codecDtype: f32, f16 или bf16", "arg");
  if (quant && quant !== "q8") fail("в браузере для Qwen3-TTS доступно квантование q8 (q4 — только в python -m vox.build)", "unsupported");
  const src = new QwenHfSource();
  src.onProgress = opts.onProgress ? (f, a, b, l) => opts.onProgress(f, a, b, l) : null;
  const tally = src._tally;

  status("читаю config.json…");
  const cfg = JSON.parse(await fs.text("config.json"));
  const tcfg = cfg.talker_config || {}, pcfg = tcfg.code_predictor_config || {};
  const tc = qMerged(QWEN_TALKER_DEFAULTS, Object.fromEntries(Object.entries(tcfg).filter(([k]) => k !== "code_predictor_config")));
  const pc = qMerged(QWEN_PRED_DEFAULTS, pcfg);
  if (pc.num_code_groups == null) pc.num_code_groups = tc.num_code_groups;
  if (!tc.head_dim) tc.head_dim = Math.trunc(tc.hidden_size / tc.num_attention_heads);
  tc.codec_language_id = tcfg.codec_language_id || {};
  const scfg = JSON.parse(await fs.text("speech_tokenizer/config.json"));
  const cc = qMerged(QWEN_CODEC_DEFAULTS, scfg.decoder_config || {});
  if (cc.codebook_dim == null) fail("speech_tokenizer/config.json: нет decoder_config.codebook_dim", "format");
  const idm = qMerged(QWEN_IDS_DEFAULTS, cfg);
  const ids = { im_start: idm.im_start_token_id, im_end: idm.im_end_token_id, tts_pad: idm.tts_pad_token_id, tts_bos: idm.tts_bos_token_id, tts_eos: idm.tts_eos_token_id };

  // --- токенизатор (как tokenizer_blobs) ---
  status("токенизатор…");
  const vocab = JSON.parse(await fs.text("vocab.json"));
  const toks = new Map();
  for (const t of Object.keys(vocab)) toks.set(Number(vocab[t]), t);
  const specials = [];
  if (fs.has("tokenizer_config.json")) {
    const added = JSON.parse(await fs.text("tokenizer_config.json")).added_tokens_decoder || {};
    for (const i of Object.keys(added)) { toks.set(Number(i), added[i].content); if (added[i].special) specials.push(added[i].content); }
  }
  const merges = (await fs.text("merges.txt")).split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l.trim() && !l.startsWith("#version"));
  const vtxt = Array.from(toks.keys()).sort((a, b) => a - b).map((i) => i + "\t" + toks.get(i)).join("\n");
  const enc = new TextEncoder();
  src.pre.set("tokenizer.vocab", enc.encode(vtxt));
  src.pre.set("tokenizer.merges", enc.encode(merges.join("\n")));
  for (const n of ["tokenizer.vocab", "tokenizer.merges"]) src.table.set(n, { dtype: "U8", shape: [src.pre.get(n).length], nbytes: src.pre.get(n).length, offset: 0 });

  // --- заголовки safetensors ---
  status("читаю заголовки safetensors…");
  const mainTensors = new Map();
  if (fs.has("model.safetensors.index.json")) {
    const wm = JSON.parse(await fs.text("model.safetensors.index.json")).weight_map;
    for (const shard of Array.from(new Set(Object.values(wm)))) {
      const f = await fs.file(shard, tally);
      for (const [k, v] of await readStHeader(f)) mainTensors.set(k, v);
    }
  } else {
    for (const [k, v] of await readStHeader(await fs.file("model.safetensors", tally))) mainTensors.set(k, v);
  }
  const codecTensors = await readStHeader(await fs.file("speech_tokenizer/model.safetensors", tally));
  src.fetched = 0;   // чтение заголовков в прогресс не входит
  src.net = fs.net || null;

  // --- описания блобов ---
  let lazy = false;   // тензоры создания голоса: в таблице есть, но в прогресс загрузки не входят и заранее не качаются
  const add = (name, dt, shape, srcBytes, produce, extra) => {
    const nb = shape.reduce((a, c) => a * c, 1) * ({ F32: 4, F16: 2, BF16: 2, I8: 1, U8: 1 }[dt]);
    src.table.set(name, Object.assign({ dtype: dt, shape, nbytes: nb, offset: 0, produce, srcBytes }, extra || {}));
    if (!lazy) src.total += srcBytes;
  };
  const readRaw = async (t) => { const r = await t.file.read(t.a, t.b); tally(r.length); return r; };
  const readF32 = async (t) => rawToF32(await readRaw(t), t.dtype, t.shape.reduce((a, c) => a * c, 1));
  const need = (t, name) => { if (!["F32", "F16", "BF16"].includes(t.dtype)) fail("тензор " + name + ": тип " + t.dtype + " не поддерживается", "format"); };
  const own = (u8) => { if ((u8.byteOffset & 3) === 0) return u8; return u8.slice(); };

  const vec = (name, t) => { need(t, name); add(name, "F32", t.shape, t.b - t.a, async () => own(convertRaw(await readRaw(t), t.dtype, "F32"))); };
  const mat = (name, t, dt, q8) => {
    need(t, name);
    const [rows, cols] = t.shape;
    if (q8 && t.shape.length === 2 && (cols & 31) === 0) {
      const sname = name + ".scale";
      add(name, "I8", [rows, cols], t.b - t.a, async (S) => {
        const w = await readF32(t), { q, s } = quantizeQ8(w, rows, cols);
        const sb = new Uint8Array(s.buffer, s.byteOffset, s.byteLength);
        S.done.set(sname, sb);
        return new Uint8Array(q.buffer, q.byteOffset, q.byteLength);
      }, { companion: sname });
      add(sname, "F32", [rows, cols >> 5], 0, null, { owner: name });
      return;
    }
    add(name, dt, t.shape, t.b - t.a, async () => own(convertRaw(await readRaw(t), t.dtype, dt)));
  };
  const matX = (name, shape, srcBytes, dt, fill) => {   // матрица, считаемая из f32 (преобразования свёрток, кодбуки)
    add(name, dt, shape, srcBytes, async () => own(f32ToRaw(await fill(), dt)));
  };

  // talker.*
  for (const [name, t] of mainTensors) {
    if (!name.startsWith("talker.")) continue;
    if (t.shape.length === 1) vec(name, t);
    else if (name.endsWith("_embedding.weight") || name.includes(".codec_embedding.") || name.includes("text_embedding")) mat(name, t, dtype, false);
    else mat(name, t, dtype, quant === "q8" && QWEN_LINEAR_Q8.test(name));
  }

  // codec: decoder.* из speech_tokenizer/model.safetensors
  const doneCb = new Set();
  for (const [name, t] of codecTensors) {
    if (!name.startsWith("decoder.")) continue;
    const key = name.slice("decoder.".length), out = "codec." + key, shape = t.shape;
    let m;
    if ((m = /quantizer\..*\.vq\.layers\.\d+\._codebook\.(cluster_usage|embedding_sum)$/.exec(key))) {
      const base = key.slice(0, key.lastIndexOf("."));
      if (doneCb.has(base)) continue;
      doneCb.add(base);
      const es = codecTensors.get("decoder." + base + ".embedding_sum"), cu = codecTensors.get("decoder." + base + ".cluster_usage");
      if (!es || !cu) fail("в кодеке нет пары embedding_sum / cluster_usage: " + base, "format");
      need(es, base); need(cu, base);
      const [rows, cols] = es.shape;
      matX("codec." + base + ".embedding", [rows, cols], (es.b - es.a) + (cu.b - cu.a), codecDtype, async () => {
        const e = await readF32(es), u = await readF32(cu);
        for (let r = 0; r < rows; r++) { const d = Math.max(u[r], f32r(1e-5)); for (let c = 0; c < cols; c++) e[r * cols + c] = f32r(e[r * cols + c] / d); }
        return e;
      });
      continue;
    }
    if (/quantizer\..*(input_proj|project_out)/.test(key)) continue;
    need(t, name);
    const bytes = t.b - t.a;
    if (/quantizer\..*output_proj\.weight$/.test(key)) {
      matX(out, [shape[0], shape[1]], bytes, codecDtype, async () => readF32(t));   // [a,b,1] → [a,b]: данные те же
    } else if (/(^upsample\.\d+\.0\.conv\.weight$)|(^decoder\.\d+\.block\.1\.conv\.weight$)/.test(key)) {
      const [cin, cout, K] = shape;   // ConvTranspose1d [cin, cout, K] → [K*cout, cin]
      matX(out, [K * cout, cin], bytes, codecDtype, async () => {
        const w = await readF32(t), o = new Float32Array(w.length);
        for (let ci = 0; ci < cin; ci++) for (let co = 0; co < cout; co++) for (let k = 0; k < K; k++) o[(k * cout + co) * cin + ci] = w[(ci * cout + co) * K + k];
        return o;
      });
    } else if (key.includes("dwconv.conv.weight")) {
      add(out, "F32", [shape[0], shape[2]], bytes, async () => own(convertRaw(await readRaw(t), t.dtype, "F32")));   // [C,1,K] → [C,K]
    } else if (key.endsWith(".weight") && shape.length === 3) {
      const [cout, cin, K] = shape;   // Conv1d [cout, cin, K] → [cout, K*cin]
      matX(out, [cout, K * cin], bytes, codecDtype, async () => {
        const w = await readF32(t), o = new Float32Array(w.length);
        for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) for (let k = 0; k < K; k++) o[co * K * cin + k * cin + ci] = w[(co * cin + ci) * K + k];
        return o;
      });
    } else if (shape.length === 1) vec(out, t);
    else if (shape.length === 2 && key.endsWith(".weight")) mat(out, t, codecDtype, false);
    else vec(out, t);
  }

  // создание голоса в браузере: энкодер диктора (ECAPA-TDNN, из основного чекпойнта) и энкодер Mimi (из speech_tokenizer)
  const voice = {};
  const kmajor = (w, cout, cin, K) => { const o = new Float32Array(w.length); for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) for (let k = 0; k < K; k++) o[co * K * cin + k * cin + ci] = w[(co * cin + ci) * K + k]; return o; };
  lazy = true;
  let hasSpk = false, hasEnc = false;
  for (const [name, t] of mainTensors) {
    if (!name.startsWith("speaker_encoder.")) continue;
    hasSpk = true;
    const out = "spk." + name.slice("speaker_encoder.".length);
    if (t.shape.length === 3) matX(out, [t.shape[0], t.shape[1] * t.shape[2]], t.b - t.a, "F32", async () => kmajor(await readF32(t), t.shape[0], t.shape[1], t.shape[2]));
    else vec(out, t);
  }
  if (hasSpk) {
    const fb = slaneyMelBasis(24000, 1024, 128, 0, 12000);
    add("spk.mel_basis", "F32", [128, 513], 0, async () => new Uint8Array(fb.buffer));
    voice.speaker_encoder = qMerged(QWEN_SPK_DEFAULTS, cfg.speaker_encoder_config || {});
  }
  const doneEnc = new Set();
  for (const [name, t] of codecTensors) {
    if (!name.startsWith("encoder.")) continue;
    const key = name.slice("encoder.".length);
    if (key.endsWith(".initialized")) continue;
    hasEnc = true;
    const m = /^(quantizer\..*\.layers\.\d+\.codebook)\.(cluster_usage|embed_sum)$/.exec(key);
    if (m) {
      const base = m[1];
      if (doneEnc.has(base)) continue;
      doneEnc.add(base);
      const es = codecTensors.get("encoder." + base + ".embed_sum"), cu = codecTensors.get("encoder." + base + ".cluster_usage");
      if (!es || !cu) fail("в энкодере кодека нет пары embed_sum / cluster_usage: " + base, "format");
      const [rows, cols] = es.shape;
      matX("enc." + base + ".embed", [rows, cols], (es.b - es.a) + (cu.b - cu.a), "F32", async () => {
        const e = await readF32(es), u = await readF32(cu);
        for (let r = 0; r < rows; r++) { const d = Math.max(u[r], f32r(1e-5)); for (let c = 0; c < cols; c++) e[r * cols + c] = f32r(e[r * cols + c] / d); }
        return e;
      });
      continue;
    }
    const out = "enc." + key;
    if (t.shape.length === 3) matX(out, [t.shape[0], t.shape[1] * t.shape[2]], t.b - t.a, "F32", async () => kmajor(await readF32(t), t.shape[0], t.shape[1], t.shape[2]));
    else if (t.shape.length === 2) mat(out, t, "F32", false);
    else vec(out, t);
  }
  if (hasEnc) {
    const ec = qMerged(QWEN_MIMI_DEFAULTS, Object.fromEntries(Object.entries(scfg.encoder_config || {}).filter(([k]) => QWEN_MIMI_KEYS.includes(k))));
    if (!ec.head_dim) ec.head_dim = Math.trunc(ec.hidden_size / ec.num_attention_heads);
    ec.valid_quantizers = scfg.encoder_valid_num_quantizers != null ? scfg.encoder_valid_num_quantizers : tc.num_code_groups;
    ec.downsample_rate = scfg.encode_downsample_rate != null ? scfg.encode_downsample_rate : 1920;
    voice.encoder = ec;
  }
  lazy = false;

  // --- манифест ---
  const gen = {};
  const g = fs.has("generate_config.json") ? JSON.parse(await fs.text("generate_config.json")) : {};
  for (const [a, b] of [["temperature", "temperature"], ["top_k", "top_k"], ["top_p", "top_p"], ["repetition_penalty", "repetition_penalty"],
    ["subtalker_temperature", "sub_temperature"], ["subtalker_top_k", "sub_top_k"], ["subtalker_top_p", "sub_top_p"], ["max_new_tokens", "max_frames"]]) if (a in g) gen[b] = g[a];
  if (gen.chunk_frames == null) gen.chunk_frames = 4;
  const sr = scfg.output_sample_rate || 24000;
  const spf = [...cc.upsample_rates, ...cc.upsampling_ratios].reduce((a, b) => a * b, 1);
  const variant = ["tokenizer_type", "tts_model_size", "tts_model_type"].filter((k) => cfg[k]).map((k) => String(cfg[k])).join("-").toLowerCase() || repoName;
  src.manifest = {
    format: "1.0",
    model: { family: "qwen3-tts", variant, type: cfg.tts_model_type || "", weights_sha256: "" },
    target: { backend: "cpu", arch: "any", features: [] },
    audio: { sample_rate: sr, samples_per_frame: spf, frame_rate: sr / spf, num_codebooks: tc.num_code_groups, codebook_size: pc.vocab_size },
    stages: [{ name: "talker", kind: "autoregressive" }, { name: "code_predictor", kind: "subloop" }, { name: "decoder", kind: "streaming_conv", context: "left_only" }],
    precision: { weights: dtype.toLowerCase(), quant: quant || "none", codec: codecDtype.toLowerCase() },
    generation: gen,
    tokenizer: { specials },
    qwen3: Object.assign({ talker: tc, predictor: pc, codec: cc, ids }, voice),
    blobs: {},
  };
  for (const [n, e] of src.table) src.manifest.blobs[n] = { dtype: e.dtype, shape: e.shape, offset: 0, nbytes: e.nbytes };
  return src;
}

// ---------- файловые системы для openQwenSource ----------
// репозиторий HF (или совместимый хост); paths — список файлов репозитория
function qwenFsFromHf(ref, paths, opts) {
  opts = Object.assign({}, opts, { _net: { requests: 0, bytes: 0, ms: 0, redirects: 0, cached: 0 } });
  const set = new Set(paths), cache = new Map();
  return {
    net: opts._net,
    has: (p) => set.has(p),
    async text(p) {
      if (!set.has(p)) fail("в репозитории " + ref.repo + " нет файла " + p + ": это не чекпойнт Qwen3-TTS?", "format");
      const url = hfUrl(ref, p, opts), r = await fetch(url, { headers: hfHeaders(opts) });
      if (r.status === 401 || r.status === 403) fail("Hugging Face: доступ к " + ref.repo + " закрыт (нужен токен: opts.token)", "network");
      if (!r.ok) fail("не удалось скачать " + p + ": HTTP " + r.status, "network");
      return r.text();
    },
    async file(p, tally) {
      if (!set.has(p)) fail("в репозитории " + ref.repo + " нет файла " + p, "format");
      if (!cache.has(p)) cache.set(p, RangeFile.fromUrl(p, hfUrl(ref, p, opts), opts, tally));
      return cache.get(p);
    },
  };
}
// словарь { "config.json": File | Blob | Uint8Array | строка-URL, "speech_tokenizer/model.safetensors": … }
function qwenFsFromMap(map, opts) {
  opts = Object.assign({}, opts, { _net: { requests: 0, bytes: 0, ms: 0, redirects: 0, cached: 0 } });
  const get = (p) => map[p];
  const net = opts._net;
  const toBytes = async (v) => (typeof v === "string" && !/^https?:|^\/|^\.\//.test(v) ? new TextEncoder().encode(v)
    : v instanceof Uint8Array ? v : typeof Blob !== "undefined" && v instanceof Blob ? new Uint8Array(await v.arrayBuffer()) : fetchBytes(v));
  return {
    net,
    has: (p) => get(p) != null,
    async text(p) { if (get(p) == null) fail("нет файла " + p, "format"); return new TextDecoder().decode(await toBytes(get(p))); },
    async file(p, tally) {
      const v = get(p);
      if (v == null) fail("нет файла " + p, "format");
      if (v instanceof Uint8Array) return RangeFile.fromBytes(p, v);
      if (typeof Blob !== "undefined" && v instanceof Blob) return RangeFile.fromBlob(p, v);
      if (typeof v === "string") return RangeFile.fromUrl(p, v, opts, tally);
      fail("неподдерживаемый источник для " + p, "arg");
    },
  };
}

// Весь источник → байты .tmod (для Vox.convert: сохранить и отдавать файлом). Нужна память под всю модель.
async function qwenSourceToTmod(src, onProgress) {
  const blobs = [];
  for (const [name, e] of src.table) {
    blobs.push({ name, dtype: e.dtype, shape: e.shape.slice(), async fill(u8) { u8.set(await src.rawAsync(name)); if (!src.pre.has(name)) src.release(name); } });
  }
  // масштаб q8 берётся из пары главного тензора: порядок заполнения — по имени, ".scale" идёт после основного
  const m = Object.assign({}, src.manifest);
  delete m.blobs;
  return writeTmod(m, blobs, onProgress);
}

// Банк mel-фильтров Slaney (как slaney_mel_basis в python/vox/build.py и librosa.filters.mel(htk=False, norm="slaney"))
function slaneyMelBasis(sr, nFft, nMels, fMin, fMax) {
  const fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = 15.0, logstep = Math.log(6.4) / 27.0;
  const h2m = (f) => (f >= minLogHz ? minLogMel + Math.log(Math.max(f, 1e-10) / minLogHz) / logstep : f / fSp);
  const m2h = (m) => (m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m);
  const nb = 1 + (nFft >> 1);
  const lin = (a, b, n) => { const o = new Float64Array(n), st = (b - a) / (n - 1); for (let i = 0; i < n; i++) o[i] = a + i * st; o[n - 1] = b; return o; };
  const fft = lin(0, sr / 2.0, nb);
  const melPts = lin(h2m(fMin), h2m(fMax), nMels + 2), melF = Array.from(melPts, m2h);
  const w = new Float32Array(nMels * nb);
  for (let i = 0; i < nMels; i++) {
    const d0 = melF[i + 1] - melF[i], d1 = melF[i + 2] - melF[i + 1], enorm = 2.0 / (melF[i + 2] - melF[i]);
    for (let k = 0; k < nb; k++) {
      const lower = -(melF[i] - fft[k]) / d0, upper = (melF[i + 2] - fft[k]) / d1;
      w[i * nb + k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
  }
  return w;
}
