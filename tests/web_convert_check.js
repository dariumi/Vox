// Проверки загрузки/конвертации vox.js в Node (без GPU): HF-ссылки против мок-сервера, сверка с .tmod из Python по тензорам.
// node tests/web_convert_check.js <vox.js> <mock-url> <work-dir> <refs-dir>
const fs = require("fs");
const path = require("path");
const [, , voxPath, host, work, refs] = process.argv;
const Vox = require(path.resolve(voxPath));
const { Tmod } = Vox._internal;
const rd = (p) => new Uint8Array(fs.readFileSync(p));
let fails = 0;
const check = (ok, what) => { console.log((ok ? "ok    " : "FAIL  ") + what); if (!ok) fails++; };
const pin = JSON.parse(fs.readFileSync(path.join(path.dirname(voxPath), "pinyin.json"), "utf8"));

function sameTensors(a, b, skip = ["mel.fb"]) {
  const A = new Tmod(a), B = new Tmod(b);
  const ka = Object.keys(A.table).sort(), kb = Object.keys(B.table).sort();
  if (JSON.stringify(ka) !== JSON.stringify(kb)) return "набор тензоров отличается";
  for (const k of ka) {
    const ia = A.info(k), ib = B.info(k);
    if (ia.dtype !== ib.dtype || JSON.stringify(ia.shape) !== JSON.stringify(ib.shape)) return "форма/тип " + k;
    if (skip.includes(k)) {
      const fa = A.f32(k), fb = B.f32(k);
      let d = 0; for (let i = 0; i < fa.length; i++) d = Math.max(d, Math.abs(fa[i] - fb[i]));
      if (d > 2e-5) return "mel.fb отличается на " + d;
      continue;
    }
    const ra = A.raw(k), rb = B.raw(k);
    for (let i = 0; i < ra.length; i++) if (ra[i] !== rb[i]) return "байты " + k + " @" + i;
  }
  return "";
}
const sameCfg = (a, b) => { const x = new Tmod(a).manifest.f5, y = new Tmod(b).manifest.f5; return JSON.stringify(x) === JSON.stringify(y); };
const raises = async (fn, code, re) => { try { await fn(); return false; } catch (e) { return e instanceof Vox.VoxError && (!code || e.code === code) && (!re || re.test(e.message)); } };

(async () => {
  const o = { hfHost: host, pinyin: pin, tokenizer: "pinyin", hash: false, dtype: "bf16", dimHead: 32 };
  // --- разбор ссылок ---
  const P = (s) => Vox.hf.parse(s, host);
  check(JSON.stringify(P("hf:SWivid/F5-TTS")) === JSON.stringify({ repo: "SWivid/F5-TTS", rev: "main", path: "", kind: "repo" }), "parse hf:org/name");
  check(P("SWivid/F5-TTS").repo === "SWivid/F5-TTS", "parse org/name");
  check(P(host + "/SWivid/F5-TTS/tree/main/F5TTS_v1_Base").path === "F5TTS_v1_Base", "parse /tree/main/<dir>");
  const bl = P(host + "/SWivid/F5-TTS/blob/main/F5TTS_v1_Base/model_1250000.safetensors");
  check(bl.kind === "file" && bl.path === "F5TTS_v1_Base/model_1250000.safetensors", "parse /blob/main/<file>");
  check(P("hf:a/b@v2/x/y.tmod").rev === "v2", "parse hf:a/b@rev/path");
  check(P("assets/tiny-f5.tmod") === null && P("./models/f5") === null && P("https://example.com/m.tmod") === null, "относительные пути и чужие URL — не HF");

  // --- репозиторий: v1_Base выбирается автоматически, остальное — альтернативы ---
  let plan = null, prog = [];
  let out = await Vox.convert("hf:mock/F5-TTS", Object.assign({}, o, { onPlan: (p) => { plan = p; }, onProgress: (f, a, b, l) => prog.push(l) }));
  check(plan && plan.kind === "f5" && plan.ckpt === "F5TTS_v1_Base/model_1250000.safetensors", "HF-репозиторий: выбран F5TTS_v1_Base/model_1250000.safetensors (" + (plan && plan.ckpt) + ")");
  check(plan && plan.alternatives && plan.alternatives.length === 2, "в плане видны альтернативы: " + JSON.stringify(plan && plan.alternatives));
  check(prog.includes("convert") && prog.some((l) => /safetensors/.test(l || "")), "прогресс скачивания и конвертации приходит");
  let d = sameTensors(out, rd(path.join(refs, "py_bf16.tmod")));
  check(d === "", "репозиторий → .tmod: тензоры совпали с vox.build_f5 побайтно" + (d ? " (" + d + ")" : ""));
  check(sameCfg(out, rd(path.join(refs, "py_bf16.tmod"))), "раздел f5 в манифесте совпал (включая словарь и таблицу пиньиня)");
  check(new Tmod(out).manifest.model.variant === "F5TTS_v1_Base", "variant = имя папки");

  // --- ссылка на папку: берётся .pt ---
  out = await Vox.convert(host + "/mock/F5-TTS/tree/main/F5TTS_Base", Object.assign({}, o, { onPlan: (p) => { plan = p; } }));
  check(plan.ckpt === "F5TTS_Base/model_1200000.pt", "ссылка /tree/main/F5TTS_Base: " + plan.ckpt);
  d = sameTensors(out, rd(path.join(refs, "py_bf16.tmod")));
  check(d === "", ".pt (torch zip+pickle) → .tmod: тензоры побайтно как у Python" + (d ? " (" + d + ")" : ""));

  // --- ссылка на конкретный файл (blob) + q8 ---
  out = await Vox.convert(host + "/mock/F5-TTS/blob/main/F5TTS_v1_Base/model_1250000.safetensors", Object.assign({}, o, { dtype: "f16", quant: "q8" }));
  d = sameTensors(out, rd(path.join(refs, "py_f16q8.tmod")));
  check(d === "", "blob-ссылка + f16/q8: тензоры побайтно как у Python" + (d ? " (" + d + ")" : ""));

  // --- готовые .tmod в репозитории: q8 приоритетнее ---
  out = await Vox.convert("hf:mock/f5-tmod", { hfHost: host, onPlan: (p) => { plan = p; } });
  check(plan.kind === "tmod" && /q8/.test(plan.tmod), "в репозитории есть .tmod → берётся q8: " + plan.tmod);
  check(Buffer.compare(Buffer.from(out), fs.readFileSync(path.join(refs, "py_f16q8.tmod"))) === 0, "скачанный .tmod отдан без изменений");
  out = await Vox.convert("hf:mock/f5-tmod", { hfHost: host, file: "model.bf16.tmod" });
  check(Buffer.compare(Buffer.from(out), fs.readFileSync(path.join(refs, "py_bf16.tmod"))) === 0, "opts.file выбирает конкретный .tmod");

  // --- прямые ссылки ---
  out = await Vox.convert(host + "/mock/f5-tmod/resolve/main/model.bf16.tmod", {});
  check(Buffer.compare(Buffer.from(out), fs.readFileSync(path.join(refs, "py_bf16.tmod"))) === 0, "прямая ссылка на .tmod");
  const direct = host + "/mock/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors";
  out = await Vox.convert(direct, { pinyin: pin, tokenizer: "pinyin", hash: false, dimHead: 32, vocos: host + "/mock/vocos/resolve/main/pytorch_model.bin" });
  d = sameTensors(out, rd(path.join(refs, "py_bf16.tmod")));
  check(d === "", "прямая ссылка на чекпойнт: vocab.txt берётся рядом, Vocos — по opts.vocos" + (d ? " (" + d + ")" : ""));
  check(await raises(() => Vox.convert(host + "/mock/F5-TTS/resolve/main/F5TTS_v1_Base/vocab.txt", {}), "format"), "ссылка на vocab.txt — внятная ошибка");

  // --- объект, байты, File-подобные ---
  const vocosFile = fs.existsSync(path.join(work, "vocos", "pytorch_model.safetensors")) ? path.join(work, "vocos", "pytorch_model.safetensors") : path.join(work, "vocos", "pytorch_model.bin");   // safetensors или torch .bin
  const vt = fs.readFileSync(path.join(work, "ckpt", "vocab.txt"), "utf8");
  out = await Vox.convert({ ckpt: rd(path.join(work, "ckpt", "model.safetensors")), vocab: vt, vocos: rd(vocosFile) },
    { pinyin: pin, tokenizer: "pinyin", hash: false, dimHead: 32 });
  check(sameTensors(out, rd(path.join(refs, "py_bf16.tmod"))) === "", "объект {ckpt, vocab, vocos} из байтов");
  out = await Vox.convert(rd(path.join(work, "ckpt", "model.safetensors")), { vocab: vt, vocos: rd(vocosFile), pinyin: pin, hash: false });
  check(sameTensors(out, rd(path.join(refs, "py_bf16.tmod"))) === "", "байты чекпойнта + opts.vocab/vocos");
  check(await raises(() => Vox.convert(rd(path.join(work, "ckpt", "model.safetensors")), {}), "arg", /словарь/), "чекпойнт без словаря → понятная ошибка");
  check(await raises(() => Vox.convert(new Uint8Array(100), {}), "format"), "мусор вместо модели → ошибка формата");

  // --- приватный репозиторий, токен ---
  check(await raises(() => Vox.convert("hf:mock/private", { hfHost: host }), "network", /токен/), "закрытый репозиторий без токена → просим токен");
  out = await Vox.convert("hf:mock/private", { hfHost: host, token: "hf_test" });
  check(Buffer.compare(Buffer.from(out), fs.readFileSync(path.join(refs, "py_bf16.tmod"))) === 0, "закрытый репозиторий с токеном");
  check(await raises(() => Vox.convert("hf:mock/absent", { hfHost: host }), "network", /не найден/), "несуществующий репозиторий → ошибка сети");

  // --- Qwen3-репозиторий распознаётся ---
  // (мок-репозиторий с пустышками вместо весов: распознан как Qwen3-TTS и отвергнут понятной ошибкой; настоящая загрузка — tests/test_web_qwen.py)
  check(await raises(() => Vox.convert("hf:mock/Qwen3-TTS-12Hz-0.6B-Base", { hfHost: host }), "format", /Qwen3|speech_tokenizer/), "репозиторий Qwen3-TTS распознан (config.json + speech_tokenizer/), битый — внятная ошибка");
  // --- словарь с CRLF и padding вокодера (то, что ломало hotstone228/F5-TTS-Russian) ---
  const { splitVocab, parseVocosPadding } = Vox._internal;
  check(JSON.stringify(splitVocab(" \r\n!\r\nа\r\n")) === JSON.stringify([" ", "!", "а"]) && JSON.stringify(splitVocab("a\nb")) === JSON.stringify(["a", "b"]),
        "vocab.txt с CRLF: \\r не попадает в токены");
  const realCfg = "feature_extractor:\n  class_path: vocos.feature_extractors.MelSpectrogramFeatures\n  init_args:\n    padding: center\n" +
                  "backbone:\n  class_path: vocos.models.VocosBackbone\nhead:\n  class_path: vocos.heads.ISTFTHead\n  init_args:\n    dim: 512\n    padding: center\n";
  check(parseVocosPadding(realCfg) === "center" && parseVocosPadding(realCfg.replace(/padding: center\n$/, "padding: same\n")) === "same" &&
        parseVocosPadding("head:\n  init_args: {dim: 48, padding: same}\n") === "same" && parseVocosPadding(null) === "center" &&
        parseVocosPadding("feature_extractor:\n  init_args:\n    padding: center\nhead:\n  init_args:\n    dim: 4\n") === "same",
        "config.yaml Vocos: padding головы ISTFT (center у charactr/vocos-mel-24khz, same по умолчанию, center без конфига)");
  console.log(fails ? "FAILED: " + fails : "ALL OK");
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
