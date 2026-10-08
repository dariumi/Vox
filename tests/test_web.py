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
        try:
            from vox.build_f5 import build
        except ImportError:
            sys.path[:0] = [os.path.join(ROOT, "python")]
            from vox.build_f5 import build
        for name, kw in (("f32", dict(dtype="f32")), ("bf16", dict(dtype="bf16")), ("q8", dict(dtype="f32", quant="q8"))):
            build(os.path.join(work, "ckpt", "model.safetensors"), os.path.join(work, "ckpt", "vocab.txt"), os.path.join(work, "vocos"),
                  os.path.join(work, f"tiny_{name}.tmod"), dim_head=32, **kw)
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "build_web.py")])
    assert rc == 0, out
    stage = tempfile.mkdtemp(prefix="voxweb_srv_")
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "dist", "vox.js"), stage)
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "test", "harness.html"), stage)
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
        check(not errors, "no page errors" + ("" if not errors else ": " + "; ".join(errors[:3])))
        br.close()
    srv.shutdown()
    print("vox.js: " + ("ALL OK" if fails == 0 else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox"), os.environ.get("VOX_WEB_WORK")))
