// ============================================================================
// Откуда берётся модель: готовый .tmod, чекпойнт (safetensors / torch) по прямой ссылке, репозиторий Hugging Face
// (ссылка, hf:org/name, org/name), File/Blob/байты, либо объект {ckpt, vocab, vocos}. Всё, что не .tmod, конвертируется
// здесь же, в браузере. Результат конвертации кэшируется (Cache API), чтобы в следующий раз не скачивать и не конвертировать.
// ============================================================================
const HF_HOST = "https://huggingface.co";
const DEFAULT_VOCOS = "charactr/vocos-mel-24khz";
const CKPT_EXT = /\.(safetensors|pt|pth)$/i;

function hfHeaders(opts) { return opts && opts.token ? { Authorization: "Bearer " + opts.token } : {}; }

// "https://huggingface.co/org/name[/tree|blob|resolve/rev/path]" | "hf:org/name[@rev][/path]" | "org/name" → { repo, rev, path, kind }
function parseHfRef(s, base) {
  if (typeof s !== "string") return null;
  const host = (base || HF_HOST).replace(/\/+$/, "");
  s = s.trim();
  let m;
  if ((m = /^hf:\/?\/?([\w.-]+\/[\w.-]+)(?:@([\w.-]+))?(?:\/(.*))?$/.exec(s))) return { repo: m[1], rev: m[2] || "main", path: m[3] || "", kind: m[3] ? "path" : "repo" };
  const esc = host.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if ((m = new RegExp("^" + esc + "/([\\w.-]+/[\\w.-]+)(?:/(tree|blob|resolve)/([^/?#]+)(?:/([^?#]*))?)?/?(?:[?#].*)?$").exec(s))) {
    const k = m[2];
    return { repo: m[1], rev: m[3] || "main", path: decodeURIComponent(m[4] || ""), kind: k === "tree" || !k ? (m[4] ? "dir" : "repo") : "file" };
  }
  if ((m = /^huggingface\.co\/([\w.-]+\/[\w.-]+)(?:\/(tree|blob|resolve)\/([^/?#]+)(?:\/(.*))?)?$/.exec(s)))
    return parseHfRef("https://huggingface.co/" + s.slice("huggingface.co/".length), base);
  if (/^[\w-][\w.-]*\/[\w-][\w.-]*$/.test(s) && !/\.(tmod|safetensors|pt|pth|bin|json|txt|wav|mp3|html|js)$/i.test(s) && !/^(\.|~)/.test(s)) return { repo: s, rev: "main", path: "", kind: "repo" };
  return null;
}

async function hfList(ref, opts) {
  const host = (opts.hfHost || HF_HOST).replace(/\/+$/, "");
  const h = hfHeaders(opts);
  const get = async (u) => {
    const r = await fetch(u, { headers: h });
    if (r.status === 401 || r.status === 403) fail("Hugging Face: доступ к " + ref.repo + " закрыт (нужен токен: opts.token)", "network");
    if (r.status === 404) fail("Hugging Face: репозиторий или ветка не найдены: " + ref.repo + "@" + ref.rev, "network");
    if (!r.ok) fail("Hugging Face: HTTP " + r.status + " для " + u, "network");
    return r.json();
  };
  let files = null;
  try {
    const t = await get(host + "/api/models/" + ref.repo + "/tree/" + encodeURIComponent(ref.rev) + "?recursive=1&expand=0");
    if (Array.isArray(t)) files = t.filter((e) => e.type === "file").map((e) => ({ path: e.path, size: e.size || (e.lfs && e.lfs.size) || 0 }));
  } catch (e) { if (e instanceof VoxError && e.code === "network" && /закрыт|не найдены/.test(e.message)) throw e; }
  if (!files) {
    const info = await get(host + "/api/models/" + ref.repo + "/revision/" + encodeURIComponent(ref.rev));
    files = (info.siblings || []).map((s) => ({ path: s.rfilename, size: s.size || 0 }));
  }
  if (!files.length) fail("в репозитории " + ref.repo + " нет файлов", "network");
  return files;
}
const hfUrl = (ref, path, opts) => (opts.hfHost || HF_HOST).replace(/\/+$/, "") + "/" + ref.repo + "/resolve/" + encodeURIComponent(ref.rev) + "/" + path.split("/").map(encodeURIComponent).join("/");

const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
const baseOf = (p) => p.slice(p.lastIndexOf("/") + 1);

// Выбор файлов из списка репозитория. Возвращает { kind: "tmod"|"f5", ... , alternatives }
function planFromFiles(files, ref, opts) {
  const paths = files.map((f) => f.path).filter((p) => !ref.path || ref.kind === "file" || p === ref.path || p.startsWith(ref.path.replace(/\/$/, "") + "/"));
  if (ref.kind === "file") {
    const p = ref.path;
    if (/\.tmod$/i.test(p)) return { kind: "tmod", tmod: p };
    if (CKPT_EXT.test(p)) return planCkpt(files.map((f) => f.path), p, ref, opts);
    fail("не знаю, что делать с файлом " + p + ": нужен .tmod, .safetensors или .pt", "format");
  }
  // Qwen3-TTS: config.json + speech_tokenizer/
  if (paths.includes("config.json") && paths.some((p) => p.startsWith("speech_tokenizer/")) && !ref.path) return { kind: "qwen3", files: paths };
  const tm = paths.filter((p) => /\.tmod$/i.test(p));
  if (tm.length) {
    if (opts.file) { const f = tm.find((p) => baseOf(p) === opts.file || p === opts.file); if (f) return { kind: "tmod", tmod: f, alternatives: tm }; }
    const rank = (p) => { const b = baseOf(p).toLowerCase(); return /q8/.test(b) ? 0 : /bf16/.test(b) ? 1 : /f16/.test(b) ? 2 : 3; };
    const best = tm.slice().sort((a, b) => rank(a) - rank(b) || (a < b ? -1 : 1))[0];
    return { kind: "tmod", tmod: best, alternatives: tm };
  }
  const ck = paths.filter((p) => CKPT_EXT.test(p) && !/(vocos|optimizer|ema_only|\.index\.json)/i.test(p) && !/(^|\/)(pytorch_model|vocos)/i.test(p));
  if (!ck.length) fail("в " + ref.repo + " не нашлось ни .tmod, ни чекпойнта F5-TTS (.safetensors/.pt)", "format");
  // группы по папкам; в каждой нужен vocab.txt
  const groups = new Map();
  for (const p of ck) { const d = dirOf(p); if (!groups.has(d)) groups.set(d, []); groups.get(d).push(p); }
  const hasVocab = (d) => paths.includes((d ? d + "/" : "") + "vocab.txt") || paths.includes("vocab.txt");
  let dirs = Array.from(groups.keys()).filter(hasVocab);
  if (!dirs.length) dirs = Array.from(groups.keys());
  const score = (d) => (/v1[_-]?base/i.test(d) ? 0 : /base/i.test(d) ? 1 : /small/i.test(d) ? 3 : 2);
  dirs.sort((a, b) => score(a) - score(b) || (a < b ? -1 : 1));
  const dir = dirs[0];
  const pick = groups.get(dir).slice().sort((a, b) => {
    const na = Number((/(\d{3,})/.exec(baseOf(a)) || [0, 0])[1]), nb = Number((/(\d{3,})/.exec(baseOf(b)) || [0, 0])[1]);
    const sa = /\.safetensors$/i.test(a) ? 0 : 1, sb = /\.safetensors$/i.test(b) ? 0 : 1;
    return sa - sb || nb - na || (a < b ? -1 : 1);
  })[0];
  return planCkpt(paths, pick, ref, opts, dirs.map((d) => d || "."));
}
function planCkpt(paths, ckpt, ref, opts, alt) {
  const d = dirOf(ckpt);
  const vocab = [(d ? d + "/" : "") + "vocab.txt", "vocab.txt"].find((p) => paths.includes(p)) || paths.find((p) => baseOf(p) === "vocab.txt") || null;
  const vocos = paths.find((p) => /(^|\/)vocos[^/]*\/(pytorch_model\.bin|model\.safetensors)$/i.test(p)) || null;
  return { kind: "f5", ckpt, vocab, vocos, variant: d ? baseOf(d) : ref.repo.split("/")[1], alternatives: alt };
}

// ---------- кэш результата конвертации ----------
const CACHE_NAME = "vox.js/models";
async function cacheGet(key) {
  try {
    if (typeof caches === "undefined") return null;
    const c = await caches.open(CACHE_NAME), r = await c.match("https://vox.cache/" + encodeURIComponent(key));
    return r ? new Uint8Array(await r.arrayBuffer()) : null;
  } catch (e) { return null; }
}
async function cachePut(key, bytes) {
  try {
    if (typeof caches === "undefined") return false;
    const c = await caches.open(CACHE_NAME);
    await c.put("https://vox.cache/" + encodeURIComponent(key), new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
    return true;
  } catch (e) { return false; }
}
async function cacheClear() { try { return typeof caches !== "undefined" && (await caches.delete(CACHE_NAME)); } catch (e) { return false; } }

// ---------- загрузка файла по любому описанию ----------
async function fetchAny(src, opts, label) {
  const p = opts.onProgress ? (f, a, b) => opts.onProgress(f, a, b, label) : null;
  if (typeof src === "string" && opts.token && (/^https:\/\/([\w-]+\.)?huggingface\.co\//.test(src) || (opts.hfHost && src.startsWith(opts.hfHost.replace(/\/+$/, "") + "/")))) {
    const r = await fetch(src, { headers: hfHeaders(opts) });
    return fetchBytes(r, p);
  }
  return fetchBytes(src, p);
}
async function fetchText(src, opts, label) {
  if (typeof src === "string" && src.includes("\n")) return src;   // это уже сам текст словаря
  return new TextDecoder().decode(await fetchAny(src, opts, label));
}

function guessTokenizer(vocabText) {
  let n = 0;
  for (const l of vocabText.split("\n")) if (/^[a-z]{1,6}[1-5]$/.test(l)) n++;
  return n >= 100 ? "pinyin" : "char";
}
let _pinyinCache = null;
async function loadPinyin(opts) {
  if (opts.pinyin === false) return null;
  if (opts.pinyin && typeof opts.pinyin === "object" && !(opts.pinyin instanceof Uint8Array)) return opts.pinyin;
  if (_pinyinCache) return _pinyinCache;
  const url = typeof opts.pinyin === "string" ? opts.pinyin : (typeof VOX_BASE_URL === "string" && VOX_BASE_URL ? VOX_BASE_URL + "pinyin.json" : null);
  if (!url) return null;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    _pinyinCache = await r.json();
    return _pinyinCache;
  } catch (e) {
    if (opts.onStatus) opts.onStatus("таблица пиньиня недоступна (" + url + "): китайский текст не будет озвучиваться");
    return null;
  }
}

// Главная функция: что угодно → байты .tmod
async function acquireTmod(src, opts = {}, keepSource = false) {
  const r = await acquireTmodRaw(src, opts);
  if (r && r.qwenSource && !keepSource) return qwenSourceToTmod(r.qwenSource, (f) => { if (opts.onProgress) opts.onProgress(f, 0, 0, "convert"); });
  return r;
}
async function acquireTmodRaw(src, opts = {}) {
  const status = (s) => { if (opts.onStatus) opts.onStatus(s); };
  const cvtOpts = () => ({ dtype: opts.dtype || "bf16", quant: opts.quant || "", tokenizer: opts.tokenizer, hash: opts.hash, modelCfg: opts.modelCfg, textMaskPadding: opts.textMaskPadding, peAttnHead: opts.peAttnHead, vocosPadding: opts.vocosPadding, vocosDtype: opts.vocosDtype });
  const cacheKeyFor = (parts) => "tmod/3|" + JSON.stringify(parts) + "|" + JSON.stringify(cvtOpts());   // 3: Vocos in f32 since 0.14.1
  const useCache = opts.cache !== false;
  // config.yaml вокодера (ради padding головы ISTFT); не нашёлся — конвертер считает, что это charactr/vocos-mel-24khz (center)
  const vocosConfigAt = async (url) => { try { return await fetchText(url, Object.assign({}, opts, { onProgress: null }), "config.yaml"); } catch (e) { return null; } };
  let vocosConfig = opts.vocosConfig != null ? opts.vocosConfig : null;
  const convertF5From = async (ckptBytes, ckptName, vocabText, vocosBytes, vocosName, variant) => {
    vocabText = splitVocab(vocabText).join("\n") + "\n";
    status("конвертация весов в формат vox (в браузере)…");
    const tokenizer = opts.tokenizer || guessTokenizer(vocabText);
    const pinyin = tokenizer === "pinyin" ? await loadPinyin(opts) : null;
    return convertF5({
      ckpt: ckptBytes, ckptName, vocab: vocabText, vocos: vocosBytes, vocosName, dtype: opts.dtype || "bf16", quant: opts.quant || "",
      tokenizer, variant: opts.variant || variant, vocosConfig, vocosPadding: opts.vocosPadding, vocosDtype: opts.vocosDtype, modelCfg: opts.modelCfg, textMaskPadding: opts.textMaskPadding, peAttnHead: opts.peAttnHead, pinyin, dimHead: opts.dimHead, mel: opts.mel, hash: opts.hash, onWarn: status,
      onProgress: (st, f) => { if (opts.onProgress && st === "convert") opts.onProgress(f, 0, 0, "convert"); },
    });
  };
  const wrapBytes = async (bytes, label, extra) => {
    const kind = sniffKind(bytes);
    if (kind === "tmod") return bytes;
    if (kind === "safetensors" || kind === "torch") {
      const vocabSrc = extra && extra.vocab != null ? extra.vocab : opts.vocab;
      if (vocabSrc == null) fail("это чекпойнт, а не .tmod: укажите словарь (opts.vocab — ссылка, текст или файл vocab.txt)", "arg");
      const vocabText = vocabSrc instanceof Uint8Array || vocabSrc instanceof ArrayBuffer || (typeof Blob !== "undefined" && vocabSrc instanceof Blob)
        ? new TextDecoder().decode(await fetchBytes(vocabSrc)) : await fetchText(vocabSrc, opts, "vocab.txt");
      let vocosBytes;
      const vs = extra && extra.vocos != null ? extra.vocos : opts.vocos;
      if (vs != null) vocosBytes = await fetchAny(typeof vs === "string" && parseHfRef(vs, opts.hfHost) && !/^https?:/.test(vs) ? vs : vs, opts, "vocos");
      else {
        const ref = { repo: DEFAULT_VOCOS, rev: "main", path: "pytorch_model.bin", kind: "file" };
        status("скачиваю вокодер Vocos (" + DEFAULT_VOCOS + ")…");
        vocosBytes = await fetchAny(hfUrl(ref, "pytorch_model.bin", opts), opts, "vocos");
        if (vocosConfig == null && !opts.vocosPadding) vocosConfig = await vocosConfigAt(hfUrl(ref, "config.yaml", opts));
      }
      return convertF5From(bytes, opts.ckptName || label || "model.safetensors", vocabText, vocosBytes, opts.vocosName || "pytorch_model.bin", opts.variant || "f5-tts");
    }
    fail("неизвестный формат файла: ни .tmod, ни safetensors, ни torch", "format");
  };

  // 1) объект {ckpt, vocab, vocos}
  if (src && typeof src === "object" && !(src instanceof Uint8Array) && !(src instanceof ArrayBuffer) && !(typeof Blob !== "undefined" && src instanceof Blob) && !(typeof Response !== "undefined" && src instanceof Response)) {
    if (src.files && typeof src.files === "object" && !src.ckpt) {   // папка Qwen3-TTS: { "config.json": File, "model.safetensors": File, … }
      status("Qwen3-TTS: конвертирую веса по тензорам (в браузере)…");
      return { qwenSource: await openQwenSource(qwenFsFromMap(src.files, opts), src.name || "qwen3-tts", opts) };
    }
    if (!src.ckpt) fail("объект-источник должен содержать ckpt (и vocab, vocos)", "arg");
    status("загрузка чекпойнта…");
    const cb = await fetchAny(src.ckpt, opts, "ckpt");
    return wrapBytes(cb, (src.ckpt && src.ckpt.name) || (typeof src.ckpt === "string" ? baseOf(src.ckpt) : "model.safetensors"), { vocab: src.vocab, vocos: src.vocos });
  }
  // 2) строка: HF или прямая ссылка
  if (typeof src === "string") {
    const ref = parseHfRef(src, opts.hfHost);
    if (ref) {
      const key = cacheKeyFor(["hf", ref.repo, ref.rev, ref.path]);
      if (useCache && !opts.refresh) { const c = await cacheGet(key); if (c) { status("модель найдена в кэше браузера"); return c; } }
      status("Hugging Face: читаю список файлов " + ref.repo + "…");
      const files = await hfList(ref, opts);
      const plan = planFromFiles(files, ref, opts);
      if (opts.onPlan) opts.onPlan(plan);
      let out;
      if (plan.kind === "tmod") {
        status("скачиваю " + plan.tmod + "…");
        out = await fetchAny(hfUrl(ref, plan.tmod, opts), opts, plan.tmod);
      } else if (plan.kind === "f5") {
        if (!plan.vocab) fail("в репозитории нет vocab.txt рядом с чекпойнтом " + plan.ckpt + " (можно передать opts.vocab)", "format");
        status("скачиваю " + plan.ckpt + "…");
        const cb = await fetchAny(hfUrl(ref, plan.ckpt, opts), opts, plan.ckpt);
        const vt = opts.vocab != null ? await fetchText(opts.vocab, opts, "vocab.txt") : await fetchText(hfUrl(ref, plan.vocab, opts), opts, "vocab.txt");
        let vb;
        if (opts.vocos != null) vb = await fetchAny(opts.vocos, opts, "vocos");
        else if (plan.vocos) {
          vb = await fetchAny(hfUrl(ref, plan.vocos, opts), opts, "vocos");
          if (vocosConfig == null && !opts.vocosPadding) vocosConfig = await vocosConfigAt(hfUrl(ref, (dirOf(plan.vocos) ? dirOf(plan.vocos) + "/" : "") + "config.yaml", opts));
        } else {
          status("скачиваю вокодер Vocos (" + DEFAULT_VOCOS + ")…");
          vb = await fetchAny(hfUrl({ repo: DEFAULT_VOCOS, rev: "main" }, "pytorch_model.bin", opts), opts, "vocos");
          if (vocosConfig == null && !opts.vocosPadding) vocosConfig = await vocosConfigAt(hfUrl({ repo: DEFAULT_VOCOS, rev: "main" }, "config.yaml", opts));
        }
        out = await convertF5From(cb, baseOf(plan.ckpt), vt, vb, opts.vocosName || (plan.vocos ? baseOf(plan.vocos) : "pytorch_model.bin"), plan.variant);
      } else if (plan.kind === "qwen3") {
        status("Qwen3-TTS: загружаю и конвертирую веса по тензорам (в браузере)…");
        return { qwenSource: await openQwenSource(qwenFsFromHf(ref, plan.files, opts), ref.repo.split("/")[1], opts) };
      }
      if (useCache) await cachePut(key, out);
      return out;
    }
    // прямая ссылка
    const key = cacheKeyFor(["url", src]);
    if (useCache && !opts.refresh && !/\.tmod(\?|#|$)/i.test(src)) { const c = await cacheGet(key); if (c) { status("модель найдена в кэше браузера"); return c; } }
    status("скачиваю " + src + "…");
    const bytes = await fetchAny(src, opts, baseOf(src.split(/[?#]/)[0]));
    const kind = sniffKind(bytes);
    if (kind === "tmod") return bytes;
    let vocab = opts.vocab;
    if (vocab == null && /^https?:/.test(src)) vocab = src.slice(0, src.lastIndexOf("/") + 1) + "vocab.txt";   // vocab.txt рядом с чекпойнтом
    const out = await wrapBytes(bytes, baseOf(src.split(/[?#]/)[0]), { vocab });
    if (useCache) await cachePut(key, out);
    return out;
  }
  // 3) байты / Blob / File / Response
  const isFile = typeof File !== "undefined" && src instanceof File;
  const bytes = await fetchAny(src, opts, isFile ? src.name : "model");
  return wrapBytes(bytes, isFile ? src.name : "model.safetensors", {});
}
