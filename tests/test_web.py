"""vox.js (WebGPU в Chromium) против ядра vox и официального F5-TTS на крошечной модели со случайными весами.
Запуск: python tests/test_web.py [build/vox]   Нужны: playwright + chromium (в песочнице WebGPU идёт через SwiftShader),
torch, f5-tts, vocos (чтобы собрать крошечную модель, как в test_f5_tiny.py).

Что проверяется:
  1. JS-токенизатор = токенизатор ядра (список символов для набора фраз);
  2. mel образца в JS = mel ядра;
  3. сквозной звук (нарезка текста, длительность, кроссфейд, громкость) против ЭТАЛОНА официального infer_batch_process
     в четырёх сценариях, шум подставлен общий;
  4. собственный шум JS (Philox) даёт ровно то же аудио, что `vox say --seed` в ядре;
  5. модели bf16 и q8 в браузере совпадают с ядром в пределах квантования.
Что НЕ проверяется: скорость на реальном GPU и качество на настоящих весах."""
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
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader",
         "--use-vulkan=swiftshader", "--disable-vulkan-surface", "--ignore-gpu-blocklist", "--no-sandbox"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def read_wav_f32(path):
    b = open(path, "rb").read()
    i, fmt = 12, None
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


def main(cli, work=None):
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        print("SKIP: needs playwright + chromium:", e)
        return 0
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails += 1

    if work is None:
        try:
            import f5_tts, torch, vocos  # noqa: F401,E401
        except Exception as e:  # noqa: BLE001
            print("SKIP: needs torch, f5-tts, vocos to build the tiny model:", e)
            return 0
        work = tempfile.mkdtemp(prefix="voxweb_")
        rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_f5.py"), "--out", work])
        assert rc == 0, out[-2000:]
        sys.path.insert(0, os.path.join(ROOT, "python"))
        from vox.build_f5 import build
        for name, kw in (("f32", dict(dtype="f32")), ("bf16", dict(dtype="bf16")), ("q8", dict(dtype="f32", quant="q8"))):
            build(os.path.join(work, "ckpt", "model.safetensors"), os.path.join(work, "ckpt", "vocab.txt"), os.path.join(work, "vocos"),
                  os.path.join(work, f"tiny_{name}.tmod"), dim_head=32, **kw)
    # F5TTS_Base (v0): text_mask_padding=False, RoPE only on the first head (pe_attn_head=1). Its weights look the same as v1,
    # only the config differs, and a wrong config gives noise on real checkpoints.
    has_v0 = False
    if not os.path.exists(os.path.join(work, "tiny_v0.tmod")):
        v0dir = os.path.join(work, "v0")
        rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_f5.py"), "--out", v0dir, "--v0", "--seed", "7", "--vocos-center", "--crlf-vocab"])
        if rc == 0:
            sys.path.insert(0, os.path.join(ROOT, "python"))
            from vox.build_f5 import build as _build
            _build(os.path.join(v0dir, "ckpt", "model.safetensors"), os.path.join(v0dir, "ckpt", "vocab.txt"), os.path.join(v0dir, "vocos"),
                   os.path.join(work, "tiny_v0.tmod"), dtype="f32", dim_head=32, model_cfg="v0")
    has_v0 = os.path.exists(os.path.join(work, "tiny_v0.tmod"))
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "build_web.py")])
    assert rc == 0, out
    stage = tempfile.mkdtemp(prefix="voxweb_srv_")
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "dist", "vox.js"), stage)
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "test", "harness.html"), stage)
    if not os.path.exists(os.path.join(ROOT, "webgpu", "web", "dist", "pinyin.json")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "gen_pinyin.py")])
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "dist", "pinyin.json"), stage)
    for fn in os.listdir(work):
        if fn.endswith((".tmod", ".safetensors", ".wav", ".json")):
            shutil.copy(os.path.join(work, fn), stage)
    e2e = json.load(open(os.path.join(work, "e2e.json")))

    handler = functools.partial(Quiet, directory=stage)
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/"

    with sync_playwright() as p:
        br = p.chromium.launch(args=FLAGS, headless=True)
        pg = br.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: print("  [console]", m.text) if m.type in ("error", "warning") else None)
        pg.goto(base + "harness.html")
        if not pg.evaluate("!!navigator.gpu"):
            print("SKIP: this Chromium has no WebGPU (SwiftShader flags not accepted)")
            br.close()
            return 0
        r = pg.evaluate("loadModel('tiny_f32.tmod')")
        check(r["info"]["family"] == "f5-tts", f"model loaded in {r['ms']:.0f} ms on {r['info']['device'].get('architecture') or 'WebGPU'}")
        # 1. токенизатор
        toks = json.load(open(os.path.join(work, "tokens.json")))
        ok, n = True, 0
        for case in toks.values():
            got = pg.evaluate("t => tokCheck(t)", case["text"])
            n += 1
            if got != case["symbols"]:
                ok = False
                print("   tokenizer mismatch:", repr(case["text"]), got, case["symbols"])
        check(ok and n > 0, f"tokenizer: {n} phrases give the same symbols as the core")
        # 2. mel образца: ядро vs JS (через дамп голоса)
        info = pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
        voice_path = os.path.join(work, "v.tvoice")
        rc, so = run([cli, "voice", "create", os.path.join(work, "ref.wav"), "--model", os.path.join(work, "tiny_f32.tmod"), "--text", e2e["ref_text"], "-o", voice_path, "--device", "cpu"])
        assert rc == 0, so
        check(info["frames"] > 10, f"voice: {info['frames']} mel frames, rms {info['rms']:.4f}")
        # 3. сценарии против эталона
        # SwiftShader — программный растеризатор, тяжёлый сценарий formula (4 куска по 1170 кадров) идёт минуты: он только при VOX_WEB_FULL=1
        for sc, prm in e2e["scenarios"].items():
            if sc == "formula" and not os.environ.get("VOX_WEB_FULL"):
                continue
            r = pg.evaluate("a => runScenario(a[0], a[1], a[2], a[3])", [sc, prm, f"noise_{sc}.safetensors", f"expected_{sc}.safetensors"])
            same = r["len"] == r["expLen"]
            check(same and r["maxDiff"] < 5e-6, f"end-to-end {sc}: {prm['chunks']} chunk(s), {r['len']} samples, max|diff| {r['maxDiff']:.2e} "
                                            f"(ref max {r['expMax']:.2e}), {r['ms']:.0f} ms")
        # 4. свой шум JS = шум ядра (через звук)
        text = "Seed test. Another sentence, with commas; and more!"
        outw = os.path.join(work, "native_seed.wav")
        rc, so = run([cli, "say", text, "--model", os.path.join(work, "tiny_f32.tmod"), "--voice", voice_path, "-o", outw, "--float", "--device", "cpu", "--seed", "5", "--nfe", "4"])
        assert rc == 0, so
        nat = read_wav_f32(outw)
        js = pg.evaluate("a => runSeed(a[0], a[1], a[2])", [text, 5, 4])
        got = np.array(js["pcm"], dtype=np.float32)
        d = float(np.abs(got[:len(nat)] - nat[:len(got)]).max()) if len(got) and len(nat) else 1.0
        check(len(got) == len(nat) and d < 5e-6, f"JS Philox noise ≡ core noise: same seed → same audio ({len(got)} samples, max|diff| {d:.2e})")
        # 4b. без WebGPU: процессорный бэкенд (JS-версии шейдеров F5-TTS: gemm, attention, ln, grn, conv1d, rope, euler…) даёт то же аудио
        pg.evaluate("loadModel('tiny_f32.tmod', { backend: 'cpu' })")
        pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
        js = pg.evaluate("a => runSeed(a[0], a[1], a[2])", [text, 5, 4])
        gotc = np.array(js["pcm"], dtype=np.float32)
        dc = float(np.abs(gotc[:len(nat)] - nat[:len(gotc)]).max()) if len(gotc) and len(nat) else 1.0
        check(len(gotc) == len(nat) and dc < 5e-5 and pg.evaluate("window.model.info.backend") == "cpu", f"процессорный бэкенд (backend:'cpu') ≡ ядро: те же {len(gotc)} отсчётов (max|diff| {dc:.2e})")
        # 4c. F5TTS_Base (v0): mask_padding=False + RoPE на первой голове; ядро, WebGPU и процессорный бэкенд дают один звук
        if has_v0:
            vv0 = os.path.join(work, "v_v0.tvoice")
            assert run([cli, "voice", "create", os.path.join(work, "ref.wav"), "--model", os.path.join(work, "tiny_v0.tmod"), "--text", e2e["ref_text"], "-o", vv0, "--device", "cpu"])[0] == 0
            outv = os.path.join(work, "native_v0.wav")
            rc, so = run([cli, "say", text, "--model", os.path.join(work, "tiny_v0.tmod"), "--voice", vv0, "-o", outv, "--float", "--device", "cpu", "--seed", "5", "--nfe", "4"])
            assert rc == 0, so
            natv = read_wav_f32(outv)
            for be in ("webgpu", "cpu"):
                info = pg.evaluate(f"loadModel('tiny_v0.tmod', {{ backend: '{be}' }})")["info"]
                pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
                jv = pg.evaluate("a => runSeed(a[0], a[1], a[2])", [text, 5, 4])
                gv = np.array(jv["pcm"], dtype=np.float32)
                dv = float(np.abs(gv[:len(natv)] - natv[:len(gv)]).max()) if len(gv) and len(natv) else 1.0
                check(len(gv) == len(natv) and dv < 5e-5 and info["f5"] == {"text_mask_padding": False, "pe_attn_head": 1},
                      f"F5TTS_Base v0 ({be}): текст без маски, RoPE на 1 голове = ядро (max|diff| {dv:.2e}), info.f5 = {info['f5']}")
            ntrue = pg.evaluate("loadModel('tiny_f32.tmod').then(r => r.info.f5)")
            check(ntrue == {"text_mask_padding": True, "pe_attn_head": None}, "v1 по умолчанию: mask_padding=true, RoPE на всех головах")
        # 5. квантованные модели
        for name in ("bf16", "q8"):
            pg.evaluate(f"loadModel('tiny_{name}.tmod')")
            pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
            outq = os.path.join(work, f"native_{name}.wav")
            vq = os.path.join(work, f"v_{name}.tvoice")
            assert run([cli, "voice", "create", os.path.join(work, "ref.wav"), "--model", os.path.join(work, f"tiny_{name}.tmod"), "--text", e2e["ref_text"], "-o", vq, "--device", "cpu"])[0] == 0
            rc, so = run([cli, "say", text, "--model", os.path.join(work, f"tiny_{name}.tmod"), "--voice", vq, "-o", outq, "--float", "--device", "cpu", "--seed", "5", "--nfe", "4"])
            assert rc == 0, so
            nat = read_wav_f32(outq)
            js = pg.evaluate("a => runSeed(a[0], a[1], a[2])", [text, 5, 4])
            got = np.array(js["pcm"], dtype=np.float32)
            rel = float(np.linalg.norm(got - nat) / np.linalg.norm(nat)) if len(got) == len(nat) else 9.9
            check(rel < 5e-3, f"{name} in the browser ≈ {name} in the core: relative L2 {rel:.2e}")
        # 6. прикладной слой: голос экспортируется/импортируется, поток, отмена, понятные ошибки
        pg.evaluate("loadModel('tiny_f32.tmod')")
        pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
        api = pg.evaluate("""async () => {
          const o = {}, m = window.model, t = 'One. Two.';
          const a = await m.synthesize(t, { voice, seed: 3, nfe_step: 2, max_chars: 4 });
          const v2 = Vox.Voice.from(JSON.parse(JSON.stringify(voice.export())));
          const b = await m.synthesize(t, { voice: v2, seed: 3, nfe_step: 2, max_chars: 4 });
          o.sameAfterExport = a.pcm.length === b.pcm.length && a.pcm.every((x, i) => x === b.pcm[i]);
          let pieces = 0; for await (const p of m.stream(t, { voice, seed: 3, nfe_step: 2, max_chars: 4 })) pieces++;
          o.pieces = pieces;
          const ac = new AbortController(); let aborted = '';
          try { await m.synthesize(t, { voice, nfe_step: 2, signal: ac.signal, onStep: () => ac.abort() }); } catch (e) { aborted = e.code; }
          o.aborted = aborted;
          const err = async (f) => { try { await f(); return ''; } catch (e) { return e.code + ':' + e.message; } };
          o.emptyText = await err(() => m.synthesize('  ', { voice }));
          o.noVoice = await err(() => m.synthesize('hi', {}));
          o.fixDur = await err(() => m.synthesize('hi', { voice, fix_duration: 3 }));
          o.badSpeed = await err(() => m.synthesize('hi', { voice, speed: 0 }));
          o.tooShort = await err(() => m.cloneVoice(new Float32Array(100), 'x', { sampleRate: 24000 }));
          o.noRate = await err(() => m.cloneVoice(new Float32Array(30000), 'x'));
          o.wav = (await Vox.wav(a.pcm, 24000).arrayBuffer()).byteLength === 44 + a.pcm.length * 2;
          o.stats = !!(m.lastRun && m.lastRun.audioSeconds > 0);
          return o; }""")
        check(api["sameAfterExport"], "Voice.export() → Voice.from() gives the same audio")
        check(api["pieces"] >= 2, f"stream() yields the audio piece by piece ({api['pieces']} pieces)")
        check(api["aborted"] == "aborted", "AbortSignal cancels synthesis")
        check(api["emptyText"].startswith("arg:") and api["noVoice"].startswith("arg:") and api["badSpeed"].startswith("arg:"), "argument errors are readable")
        check(api["fixDur"].startswith("unsupported:") and "fix_duration" in api["fixDur"], "fix_duration is rejected with a clear message")
        check(api["tooShort"].startswith("arg:") and api["noRate"].startswith("arg:"), "short reference / missing sampleRate → clear errors")
        check(api["wav"] and api["stats"], "Vox.wav() size is right, run statistics are reported")
        # 7. модель берётся по ссылке на «репозиторий HF» и конвертируется в самом браузере; второй раз — из кэша
        from mockhf import MockHF, build_repos
        tmp = tempfile.mkdtemp(prefix="voxweb_hf_")
        repos, refs = build_repos(work, tmp)
        hf = MockHF(repos)
        try:
            pg.evaluate("loadModel('tiny_f32.tmod')")
            pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
            base_out = pg.evaluate("runSeed('Short one.', 9, 2)")
            opts = {"hfHost": hf.url, "dtype": "f32", "dimHead": 32}
            r = pg.evaluate("a => loadFrom(a[0], a[1])", ["hf:mock/F5-TTS", opts])
            check(any("конвертация" in x for x in r["statuses"]) and r["info"]["variant"] == "F5TTS_v1_Base",
                  f"Vox.load('hf:mock/F5-TTS'): скачано и сконвертировано в браузере за {r['ms']:.0f} мс")
            pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
            conv = pg.evaluate("runSeed('Short one.', 9, 2)")
            a, b = np.array(base_out["pcm"], dtype=np.float32), np.array(conv["pcm"], dtype=np.float32)
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(a)) if len(a) == len(b) else 9.9
            check(rel < 1e-3, f"модель, сконвертированная в браузере, звучит как собранная Python (относительная L2 {rel:.2e})")
            keys = pg.evaluate("cacheKeys()")
            check(len(keys) == 1, "результат конвертации лёг в Cache API")
            r2 = pg.evaluate("a => loadFrom(a[0], a[1])", ["hf:mock/F5-TTS", opts])
            n_ckpt = sum(1 for p, _ in hf.log if p.endswith(".safetensors"))
            check(any("кэш" in x for x in r2["statuses"]) and n_ckpt == 1, f"повторная загрузка — из кэша браузера ({r2['ms']:.0f} мс), чекпойнт скачан {n_ckpt} раз")
            pinyin_hits = pg.evaluate("Vox.config().baseUrl")
            check(pinyin_hits.endswith("/"), "baseUrl (для pinyin.json) определён по адресу скрипта: " + pinyin_hits)
        finally:
            hf.close()
        # 8. двусторонний стриминг в браузере (та же логика коммита, что в ядре)
        pg.evaluate("loadModel('tiny_f32.tmod')")
        pg.evaluate(f"setRef('ref.wav', {json.dumps(e2e['ref_text'])})")
        dx = pg.evaluate("""async () => {
          const o = {}, m = window.model, base = { voice, seed: 3, nfe_step: 2, max_chars: 4 };
          const whole = await m.synthesize('One. Two.', base);
          const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
          const cat = (parts) => { const n = parts.reduce((s, p) => s + p.length, 0), r = new Float32Array(n); let k = 0; for (const p of parts) { r.set(p, k); k += p.length; } return r; };
          // auto: текст порциями, итог = синтез целиком (F5 озвучивает законченные куски, последний держит до конца)
          let d = m.duplex(base);
          d.write('One'); d.write('. Tw'); d.write('o.'); d.end();
          const parts = []; for await (const c of d.audio()) parts.push(c);
          o.autoSame = same(cat(parts), whole.pcm);
          o.autoCaps = d.caps;
          // manual: до end() в модель ничего не уходит
          d = m.duplex(Object.assign({ commit: 'manual' }, base));
          d.write('One. '); d.write('Two.');
          await new Promise((r) => setTimeout(r, 300));
          o.manualQuiet = d.q.length === 0 && d.info().committed_bytes === 0 && d.info().pending_bytes === 9;
          o.manualSame = same((await d.final()).pcm, whole.pcm);
          // sentence: законченное предложение озвучивается, пока второе набирается
          d = m.duplex(Object.assign({ commit: 'sentence' }, base));
          d.write('One. Tw');
          const first = await d.read(300000);
          o.sentFirst = first && first.length > 0 && d.info().pending === 'Tw' && !d.done;
          d.write('o.');
          const rest = await d.final();
          o.sentTotal = rest.pcm.length > 0 && rest.pcm.every(Number.isFinite);
          // flush: фраза озвучена без end(), конец затухает до нуля
          d = m.duplex(Object.assign({ commit: 'manual' }, base));
          d.write('One'); d.flush();
          const got = []; let c;
          while ((c = await d.read(120000)) && c.length) { got.push(c); if (d.q.length === 0 && got.length) { c = await d.read(1500); if (c) got.push(c); else break; } }
          const fl = cat(got);
          o.flushLen = fl.length; o.flushLast = fl.length ? Math.abs(fl[fl.length - 1]) : 1; o.flushNotDone = !d.done;
          d.cancel();
          // набор текста: хвост правится, озвученное — нет
          d = m.duplex(Object.assign({ commit: 'sentence' }, base));
          const r1 = d.setText('One. T');
          await d.read(300000);
          const r2 = d.setText('One. Tw');
          const r3 = d.setText('Uno. Tw');
          o.typing = [r1.ok, r1.committed, r2.ok, r2.pending, r3.ok, r3.spoken];
          d.cancel();
          // ошибки
          const err = (f) => { try { f(); return ''; } catch (e) { return e.code; } };
          d = m.duplex(base); d.end();
          o.errWrite = err(() => d.write('x'));
          o.errCommit = err(() => m.duplex(Object.assign({}, base, { commit: 'whenever' })));
          o.errVoice = err(() => m.duplex({}));
          return o; }""")
        check(dx["autoSame"] and dx["autoCaps"]["text_in"] == "chunk", "duplex auto в браузере: текст порциями → тот же звук, что у synthesize()")
        check(dx["manualQuiet"] and dx["manualSame"], "duplex manual: до end() в модель ничего не уходит, итог = синтез целиком")
        check(dx["sentFirst"] and dx["sentTotal"], "duplex sentence: первое предложение озвучено, пока второе набирается")
        check(dx["flushLen"] > 1000 and dx["flushLast"] < 1e-6 and dx["flushNotDone"], f"duplex flush: фраза озвучена без end(), конец затухает ({dx['flushLen']} отсч., |last|={dx['flushLast']:.1e})")
        check(dx["typing"] == [True, 5, True, "Tw", False, "One. "], f"duplex setText: правка хвоста принята, правка озвученного отклонена {dx['typing']}")
        check(dx["errWrite"] == "state" and dx["errCommit"] == "arg" and dx["errVoice"] == "arg", "duplex: понятные ошибки (write после end, commit, голос)")
        check(not errors, "no page errors" + ("" if not errors else ": " + "; ".join(errors[:3])))
        br.close()
    srv.shutdown()
    print("vox.js: " + ("ALL OK" if fails == 0 else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox"), os.environ.get("VOX_WEB_WORK")))
