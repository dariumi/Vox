"""Qwen3-TTS в браузере: загрузка по ссылке HF, конвертация в JS, синтез на WebGPU, двусторонний стриминг.
Запуск: python tests/test_web_qwen.py [build/vox]
Нужны: node, playwright + chromium (WebGPU через SwiftShader), torch + qwen-tts (крошечная модель из официальных классов,
tools/make_tiny_official.py), словарь Qwen2 (VOX_QWEN2_GGUF=qwen2.gguf), собранный build/vox. Иначе SKIP.

Что проверяется (для CustomVoice и VoiceDesign):
  1. конвертация чекпойнта HF в JS (bf16, f16+q8) совпадает с `python -m vox.build` побайтно, каждый тензор;
  2. Vox.load('hf:…') скачивает по HTTP Range только нужные тензоры (мок-сервер HF) и звучит так же, как ядро (`vox say --seed`);
  3. загрузка из набора файлов ({files: {…}}: File/Blob) даёт то же аудио;
  4. двусторонний стриминг: порции текста (auto) = целая фраза; manual молчит до end(); sentence озвучивает предложение,
     пока следующее набирается; пауза → новое высказывание;
  5. токенизатор JS = токенизатор ядра;
  6. Base: голос из записи в браузере (mel + ECAPA + Mimi) = `vox voice create` (вектор диктора, коды), а звук с таким голосом = `vox say --voice`.
Что НЕ проверяется: скорость на реальном GPU и время создания голоса на настоящих 0.6B/1.7B (только крошечные модели со случайными весами)."""
import functools
import http.server
import json
import os
import shutil
import socketserver
import struct
import subprocess
import sys
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "python")]
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader",
         "--use-vulkan=swiftshader", "--disable-vulkan-surface", "--ignore-gpu-blocklist", "--no-sandbox"]
TEXT = "Hello, this is a speaker test."
CASES = {"custom_voice": dict(speaker="aiden", instruct="Speak slowly and calmly"), "voice_design": dict(speaker=None, instruct="A calm, deep narrator voice")}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def read_wav_f32(path):
    b = open(path, "rb").read()
    i, fmt, raw = 12, None, b""
    while i < len(b):
        cid, n = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", b[i + 8:i + 24])
        if cid == b"data":
            raw = b[i + 8:i + 8 + n]
            break
        i += 8 + n + (n & 1)
    return np.frombuffer(raw, "<f4").astype(np.float32) if fmt[0] == 3 else np.frombuffer(raw, "<i2").astype(np.float32) / 32768


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def main(cli):
    from playwright.sync_api import sync_playwright  # noqa: E402
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails += 1

    cli = os.path.abspath(cli)
    vocab = os.environ.get("VOX_QWEN2_GGUF", "qwen2.gguf")
    if shutil.which("node") is None or not os.path.exists(vocab) or not os.path.exists(cli):
        print("SKIP: нужны node, build/vox и словарь Qwen2 (VOX_QWEN2_GGUF)")
        return 0
    try:
        import torch  # noqa: F401
        import qwen_tts  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("SKIP: нужны torch и qwen-tts, чтобы собрать крошечную модель:", e)
        return 0
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "build_web.py")])
    assert rc == 0, out
    work = tempfile.mkdtemp(prefix="voxwq_")
    hfroot = os.path.join(work, "hf", "mock")
    os.makedirs(hfroot)
    refs, models = {}, {}
    from vox.build import build
    for var, kw in CASES.items():
        d = os.path.join(hfroot, "Qwen3-" + var)
        rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_official.py"), "--vocab", vocab, "--out", d, "--variant", var])
        assert rc == 0, out[-2000:]
        models[var] = d
        r = {}
        for name, a in (("f32", dict(dtype="f32")), ("bf16", dict(dtype="bf16")), ("f16q8", dict(dtype="f16", quant="q8"))):
            r[name] = os.path.join(work, f"{var}_{name}.tmod")
            build(d, r[name], sha=False, **a)
        refs[var] = r
        # эталонное аудио ядра
        wav = os.path.join(work, f"{var}.wav")
        cmd = [cli, "say", TEXT, "--model", r["f32"], "--instruct", kw["instruct"], "--lang", "english", "--seed", "1", "--max-frames", "10", "--float", "-o", wav]
        if kw["speaker"]:
            cmd += ["--speaker", kw["speaker"]]
        rc, out = run(cmd)
        assert rc == 0, out[-1500:]
        refs[var]["wav"] = wav

    # Base: крошечная модель и запись для клонирования
    base_dir = os.path.join(hfroot, "Qwen3-base")
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_official.py"), "--vocab", vocab, "--out", base_dir, "--variant", "base"])
    assert rc == 0, out[-2000:]
    from vox.build import build as _b
    base_tmod = os.path.join(work, "base_f32.tmod")
    _b(base_dir, base_tmod, dtype="f32", sha=False)
    sr = 24000
    t = np.arange(sr * 5) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    x = sum(np.sin(k * 2 * np.pi * np.cumsum(f0) / sr) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 1.5
    x = x + 0.02 * np.random.RandomState(0).randn(len(t))
    ref_pcm = (x / np.abs(x).max() * 0.5).astype(np.float32)
    ref_wav = os.path.join(work, "ref.wav")
    import wave
    with wave.open(ref_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((ref_pcm * 32767).astype("<i2").tobytes())
    ref_pcm16 = np.frombuffer((ref_pcm * 32767).astype("<i2").tobytes(), "<i2").astype(np.float32) / 32768   # ровно то, что прочитает ядро
    ref_text = "Hello there friend"
    ref_voice = os.path.join(work, "ref_core.tvoice")
    rc, out = run([cli, "voice", "create", ref_wav, "--model", base_tmod, "--text", ref_text, "-o", ref_voice])
    assert rc == 0, out[-1500:]

    # 1. конвертация: побайтно с Python
    for var, d in models.items():
        for name, args in (("bf16", ["bf16"]), ("f16q8", ["f16", "q8"])):
            rc, out = run(["node", os.path.join(HERE, "web_qwen_check.js"), os.path.join(ROOT, "webgpu", "web", "dist", "vox.js"), d, refs[var][name]] + args)
            check(rc == 0, f"{var}: конвертация {name} в JS = python -m vox.build ({out.strip().splitlines()[-1] if out.strip() else ''})")

    # токены ядра для сверки токенизатора
    phrases = ["Hello, this is a speaker test.", "Привет, мир! Как дела?", "Numbers 12345 and 3.14, don't stop...", "你好，世界", "line one\nline two  spaced"]
    rc, out = run([cli, "tokenize", "--model", refs["custom_voice"]["f32"]], input="\n".join(json.dumps(p) for p in phrases) + "\n")
    core_tokens = [json.loads(l) for l in out.strip().splitlines()] if rc == 0 else None

    stage = tempfile.mkdtemp(prefix="voxwq_srv_")
    for f in ("vox.js",):
        shutil.copy(os.path.join(ROOT, "webgpu", "web", "dist", f), stage)
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "test", "harness.html"), stage)
    from mockhf import MockHF
    hf = MockHF(os.path.join(work, "hf"), cdn=True)   # /resolve/ отвечает редиректом на «CDN», как настоящий HF
    srv = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=stage))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            br = p.chromium.launch(args=FLAGS, headless=True)
            pg = br.new_page()
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console", lambda m: print("  [console]", m.text) if m.type == "error" else None)
            pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/harness.html")
            if not pg.evaluate("!!navigator.gpu"):
                print("SKIP: в этом Chromium нет WebGPU")
                br.close()
                return 0
            pg.set_default_timeout(1800000)
            url = f"http://127.0.0.1:{srv.server_address[1]}/harness.html"

            def fresh_eval(js, args):
                """Each block in its own page (the previous page's GPU devices are released), one retry if headless Chromium loses
                the WebGPU device mid-block ("A valid external Instance reference no longer exists" on Chromium 156 / SwiftShader):
                that is the browser's software adapter failing, not vox, and every check of the block is re-run from scratch."""
                nonlocal pg
                for attempt in range(2):
                    try:
                        pg.context.close() if pg.context is not br.contexts[0] else pg.close()
                    except Exception:  # noqa: BLE001
                        pass
                    pg = br.new_context().new_page()   # fresh context: an empty Cache API, as the checks of the block expect
                    pg.on("pageerror", lambda e: errors.append(str(e)))
                    pg.on("console", lambda m: print("  [console]", m.text) if m.type == "error" else None)
                    pg.set_default_timeout(1800000)
                    pg.goto(url)
                    try:
                        return pg.evaluate(js, args)
                    except Exception as e:  # noqa: BLE001
                        msg = str(e)
                        lost = "Instance reference" in msg or "device-lost" in msg or "Device was lost" in msg or "lost" in msg.lower()
                        if attempt == 0 and lost:
                            print(f"  note: WebGPU device lost in headless Chromium, re-running the block once ({msg.splitlines()[0][:160]})")
                            errors[:] = [x for x in errors if "lost" not in x.lower() and "Instance reference" not in x]
                            continue
                        raise

            for var, kw in CASES.items():
                ref = read_wav_f32(refs[var]["wav"])
                res = fresh_eval("""async ([host, repo, text, spk, ins]) => {
                  const o = {};
                  const t0 = performance.now(); let prog = 0;
                  const m = await Vox.load('hf:' + repo, { hfHost: host, dtype: 'f32', onProgress: (f) => { prog = f; } });
                  o.loadMs = performance.now() - t0; o.prog = prog; o.type = m.info.type; o.family = m.info.family;
                  const mk = (mm) => mm.speakerVoice(spk, ins, { language: 'english' });
                  const r = await m.synthesize(text, { voice: mk(m), seed: 1, language: 'english', max_frames: 10 });
                  o.pcm = Array.from(r.pcm);
                  // набор файлов (Blob) вместо ссылки
                  const names = ['config.json', 'generate_config.json', 'vocab.json', 'merges.txt', 'tokenizer_config.json', 'model.safetensors', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors'];
                  const files = {};
                  for (const n of names) { const rr = await fetch(host + '/' + repo + '/resolve/main/' + n); if (rr.ok) files[n] = await rr.blob(); }
                  const m2 = await Vox.load({ files }, { dtype: 'f32' });
                  const r2 = await m2.synthesize(text, { voice: mk(m2), seed: 1, language: 'english', max_frames: 10 });
                  let md = 0; for (let i = 0; i < r.pcm.length; i++) md = Math.max(md, Math.abs(r.pcm[i] - r2.pcm[i]));
                  o.filesDiff = r2.pcm.length === r.pcm.length ? md : -1;
                  // двусторонний стриминг
                  const base = { voice: mk(m), seed: 3, language: 'english', max_frames: 10 };
                  const cat = (parts) => { const n = parts.reduce((s, q) => s + q.length, 0), z = new Float32Array(n); let k = 0; for (const q of parts) { z.set(q, k); k += q.length; } return z; };
                  const diff = (a, b) => { if (a.length !== b.length) return 1e9; let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; };
                  const whole = await m.synthesize(text, base);
                  let d = m.duplex(base);
                  d.write('Hello, th'); d.write('is is a spe'); d.write('aker test.'); d.end();
                  const parts = []; for await (const c of d.audio()) parts.push(c);
                  o.autoDiff = diff(cat(parts), whole.pcm);
                  d = m.duplex(Object.assign({ commit: 'manual' }, base));
                  d.write(text);
                  await new Promise((q) => setTimeout(q, 400));
                  o.manualQuiet = d.q.length === 0 && d.info().committed_bytes === 0;
                  o.manualDiff = diff((await d.final()).pcm, whole.pcm);
                  d = m.duplex(Object.assign({ commit: 'sentence' }, base));
                  d.write('Hello there. Tw');
                  const first = await d.read(300000);
                  o.sentFirst = !!(first && first.length) && d.info().pending === 'Tw' && !d.done;
                  d.write('o.');
                  const rest = await d.final();
                  o.sentOk = rest.pcm.length > 0 && Array.prototype.every.call(rest.pcm, Number.isFinite);
                  d = m.duplex(Object.assign({ utterance_gap_ms: 400 }, base));
                  d.write('Hello there. ');
                  const u1 = await d.read(300000);
                  await new Promise((q) => setTimeout(q, 1500));
                  d.write('Second one.');
                  await d.final();
                  o.gap = { u1: !!(u1 && u1.length), n: d.be.count };
                  // кэш блоков: повторная загрузка той же ссылки идёт из Cache API без сети
                  o.net1 = m.info.net;
                  await new Promise((q) => setTimeout(q, 500));
                  const m3 = await Vox.load('hf:' + repo, { hfHost: host, dtype: 'f32' });
                  o.net2 = m3.info.net;
                  const r3 = await m3.synthesize(text, { voice: mk(m3), seed: 1, language: 'english', max_frames: 10 });
                  let d3 = 0; for (let i = 0; i < r.pcm.length; i++) d3 = Math.max(d3, Math.abs(r.pcm[i] - r3.pcm[i]));
                  o.cacheDiff = r3.pcm.length === r.pcm.length ? d3 : -1;
                  for (const mm of [m, m2, m3]) mm.dispose();   // release the GPU devices (each model has its own)
                  return o;
                }""", [hf.url, "mock/Qwen3-" + var, TEXT, kw["speaker"], kw["instruct"]])
                pcm = np.array(res["pcm"], dtype=np.float32)
                n1, n2 = res["net1"], res["net2"]
                check(n1 and 0 < n1["requests"] < 60 and n1["redirects"] == 2, f"{var}: чтение крупными блоками и после первого редиректа сразу с адреса CDN ({n1})")
                check(n2 and n2["mib"] == 0 and n2["requests"] <= 2 and n2["cachedBlocks"] > 0 and res["cacheDiff"] == 0, f"{var}: повторная загрузка из кэша блоков без скачивания (только заголовки файлов), звук тот же ({n2}, расхождение {res['cacheDiff']})")
                check(res["family"] == "qwen3-tts" and res["type"] == var, f"{var}: Vox.load('hf:…') собрал модель в браузере за {res['loadMs']:.0f} мс")
                check(len(pcm) == len(ref) and float(np.abs(pcm - ref).max()) < 1e-3 * max(1.0, float(np.abs(ref).max())),   # те же токены; допуск — шум f32 GPU/CPU в декодере со случайными весами (амплитуда ~1)
                      
                      f"{var}: звук из браузера = vox say --seed 1 (макс. расхождение {float(np.abs(pcm - ref).max()) if len(pcm) == len(ref) else 'длина'})")
                check(res["prog"] > 0.99, f"{var}: прогресс загрузки дошёл до {res['prog']:.2f}")
                check(res["filesDiff"] == 0, f"{var}: загрузка из набора файлов (Blob) = загрузка по ссылке")
                check(res["autoDiff"] < 1e-6, f"{var}: duplex auto, порции текста = целая фраза ({res['autoDiff']:.1e})")
                check(res["manualQuiet"] and res["manualDiff"] < 1e-6, f"{var}: duplex manual молчит до end() и даёт ту же запись")
                check(res["sentFirst"] and res["sentOk"], f"{var}: duplex sentence озвучивает предложение, пока следующее набирается")
                check(res["gap"]["u1"] and res["gap"]["n"] >= 2, f"{var}: пауза закрывает высказывание, дальше новое ({res['gap']})")
            # 6. Base: голос из записи в браузере
            from safetensors import safe_open
            res = fresh_eval("""async ([host, repo, pcm, text]) => {
              const m = await Vox.load('hf:' + repo, { hfHost: host, dtype: 'f32' });
              const o = { clone: m.capabilities.clone_from_audio, type: m.type };
              const prog = []; const t0 = performance.now();
              const v = await m.cloneVoice(Float32Array.from(pcm), text, { sampleRate: 24000, name: 'demo', onProgress: (p) => prog.push(p) });
              o.cloneMs = performance.now() - t0; o.prog = prog.length;
              o.emb = Array.from(v.speakerEmb); o.codes = Array.from(v.refCodes); o.frames = v.refFrames; o.cb = v.refCodebooks; o.tok = Array.from(v.refTextTokens);
              const r = await m.synthesize('Hello, this is a clone test.', { voice: v, seed: 1, language: 'english', max_frames: 10 });
              o.pcm = Array.from(r.pcm);
              o.tvoice = Array.from(v.toTvoice());
              const v2 = m.voiceFromFile(new Uint8Array(o.tvoice));
              o.reDiff = v2.speakerEmb.every((z, i) => z === v.speakerEmb[i]) && v2.refCodes.every((z, i) => z === v.refCodes[i]);
              const xv = await m.cloneVoice(Float32Array.from(pcm), '', { sampleRate: 24000, xvectorOnly: true });
              o.xvMode = xv.mode; o.xvEmb = Array.from(xv.speakerEmb);
              let err = '';
              try { await m.cloneVoice(Float32Array.from(pcm), '', { sampleRate: 24000 }); } catch (e) { err = e.code; }
              o.errNoText = err;
              m.dispose();
              return o;
            }""", [hf.url, "mock/Qwen3-base", [float(z) for z in ref_pcm16], ref_text])
            with safe_open(ref_voice, "pt") as f:
                emb_core = f.get_tensor("speaker_emb").numpy()
                codes_core = f.get_tensor("ref_codes").numpy()
                tok_core = f.get_tensor("ref_text_tokens").numpy()
            emb = np.array(res["emb"], dtype=np.float32)
            rel = float(np.linalg.norm(emb - emb_core) / np.linalg.norm(emb_core))
            check(res["clone"] and res["type"] == "base", "Base: модель предлагает клонирование по записи в браузере")
            check(rel < 1e-5, f"Base: вектор диктора из браузера = vox voice create (отн. ошибка {rel:.1e})")
            codes = np.array(res["codes"], dtype=np.int16).reshape(res["frames"], res["cb"])
            check(codes.shape == codes_core.shape and bool((codes == codes_core).all()), f"Base: коды образца из браузера = vox voice create ({codes.shape})")
            check(list(tok_core) == res["tok"], "Base: токены текста образца совпали")
            check(res["reDiff"], "Base: голос → .tvoice → голос без потерь")
            check(res["xvMode"] == "xvector_only" and len(res["xvEmb"]) == len(res["emb"]), "Base: режим только вектора диктора")
            check(res["errNoText"] == "arg", "Base: без текста образца — понятная ошибка")
            tv = os.path.join(work, "browser.tvoice")
            open(tv, "wb").write(bytes(res["tvoice"]))
            wav2 = os.path.join(work, "clone.wav")
            rc, out = run([cli, "say", "Hello, this is a clone test.", "--model", base_tmod, "--voice", tv, "--lang", "english", "--seed", "1", "--max-frames", "10", "--float", "-o", wav2])
            assert rc == 0, out[-1500:]
            ref2 = read_wav_f32(wav2)
            got = np.array(res["pcm"], dtype=np.float32)
            check(len(got) == len(ref2) and float(np.abs(got - ref2).max()) < 1e-3 * max(1.0, float(np.abs(ref2).max())),
                  f"Base: синтез с голосом из браузера = vox say --voice (расхождение {float(np.abs(got - ref2).max()) if len(got) == len(ref2) else 'длина'})")
            ranges = [x for x in hf.log if x[0].startswith("RANGE")]
            check(0 < len(ranges) < 100, f"загрузка шла крупными кусками HTTP Range, а не по запросу на тензор ({len(ranges)} запросов на 300+ тензоров)")
            if core_tokens:
                toks = fresh_eval("""async ([host, repo, phrases]) => { const m = await Vox.load('hf:' + repo, { hfHost: host, dtype: 'bf16' }); const t = phrases.map((p) => Array.from(m.tok.encode(p))); m.dispose(); return t; }""",
                                   [hf.url, "mock/Qwen3-custom_voice", phrases])
                check(toks == core_tokens, "токенизатор JS = токенизатор ядра")
            check(not errors, "ошибок страницы нет" if not errors else f"ошибки страницы: {errors[:3]}")
            br.close()
    finally:
        hf.close()
        srv.shutdown()
    print("vox.js Qwen3: " + ("всё в порядке" if not fails else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/vox"))
