"""Vox.js без WebGPU: процессорный бэкенд (JS-версии всех шейдеров, счёт порциями с уступкой потока).
Запуск: python tests/test_web_cpu.py [build/vox]. Нужны те же зависимости, что у test_web_qwen.py; иначе SKIP.
Проверяется на крошечной Qwen3-TTS (CustomVoice): в node без navigator.gpu бэкенд выбирается сам (auto → cpu);
звук = звуку ядра (`vox say --seed`) для f32, bf16 и f16; для f16+q8 (ядро квантует активации, поэтому оно не эталон)
звук = звуку WebGPU (SwiftShader) в Chromium; в Chromium процессорный бэкенд не замораживает страницу (таймер тикает во время синтеза).
Что НЕ проверяется: скорость на настоящих моделях (на CPU 0.6B заведомо медленнее реального времени) и F5-TTS на процессоре."""
import functools
import os
import shutil
import socketserver
import sys
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "python")]
import test_web_qwen as W  # noqa: E402


def main(cli):
    from playwright.sync_api import sync_playwright
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        fails += not ok

    cli = os.path.abspath(cli)
    vocab = os.environ.get("VOX_QWEN2_GGUF", "qwen2.gguf")
    if shutil.which("node") is None or not os.path.exists(vocab) or not os.path.exists(cli):
        print("SKIP: нужны node, build/vox и словарь Qwen2 (VOX_QWEN2_GGUF)")
        return 0
    try:
        import torch  # noqa: F401
        import qwen_tts  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("SKIP: нужны torch и qwen-tts:", e)
        return 0
    rc, out = W.run([sys.executable, os.path.join(ROOT, "tools", "build_web.py")])
    assert rc == 0, out
    work = tempfile.mkdtemp(prefix="voxcpu_")
    d = os.path.join(work, "model")
    rc, out = W.run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_official.py"), "--vocab", vocab, "--out", d, "--variant", "custom_voice"])
    assert rc == 0, out[-2000:]
    from vox.build import build
    spk, ins = "aiden", "Speak slowly and calmly"
    voxjs = os.path.join(ROOT, "webgpu", "web", "dist", "vox.js")
    stage = os.path.join(work, "stage")
    os.makedirs(stage)
    shutil.copy(voxjs, stage)
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "test", "harness.html"), stage)
    cpu = {}
    for name, a in (("f32", dict(dtype="f32")), ("bf16", dict(dtype="bf16")), ("f16", dict(dtype="f16")), ("f16q8", dict(dtype="f16", quant="q8"))):
        tm = os.path.join(stage, f"m_{name}.tmod")
        build(d, tm, sha=False, **a)
        o = os.path.join(work, f"cpu_{name}.f32")
        rc, out = W.run(["node", os.path.join(HERE, "web_cpu_check.js"), voxjs, tm, o, spk, ins])
        check(rc == 0, f"{name}: node без WebGPU сам выбрал cpu и синтезировал" + ("" if rc == 0 else ": " + out[-400:]))
        if rc != 0:
            continue
        cpu[name] = np.fromfile(o, "<f4")
        if name != "f16q8":
            wav = os.path.join(work, f"core_{name}.wav")
            rc, out = W.run([cli, "say", W.TEXT, "--model", tm, "--instruct", ins, "--speaker", spk, "--lang", "english", "--seed", "1", "--max-frames", "10", "--float", "-o", wav])
            assert rc == 0, out[-1500:]
            ref = W.read_wav_f32(wav)
            ok = len(ref) == len(cpu[name]) and float(np.abs(ref - cpu[name]).max()) < 1e-3 * max(1.0, float(np.abs(ref).max()))
            check(ok, f"{name}: звук на процессорном бэкенде = vox say --seed 1 (расхождение {float(np.abs(ref - cpu[name]).max()) if len(ref) == len(cpu[name]) else 'длина'})")

    srv = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(W.Quiet, directory=stage))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        br = p.chromium.launch(args=W.FLAGS, headless=True)
        pg = br.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/harness.html")
        pg.set_default_timeout(1800000)
        has_gpu = pg.evaluate("!!navigator.gpu")
        res = pg.evaluate("""async ([name, spk, ins, gpu]) => {
          const bytes = new Uint8Array(await (await fetch('m_' + name + '.tmod')).arrayBuffer());
          const run = async (backend) => {
            const m = await Vox.load(bytes.slice(), { backend });
            let ticks = 0; const iv = setInterval(() => { ticks++; }, 5);
            const r = await m.synthesize('Hello, this is a speaker test.', { voice: m.speakerVoice(spk, ins, { language: 'english' }), seed: 1, language: 'english', max_frames: 10 });
            clearInterval(iv);
            const be = m.info.backend; m.dispose();
            return { pcm: Array.from(r.pcm), ticks, be };
          };
          const o = { cpu: await run('cpu') };
          o.auto = Vox.backends();
          if (gpu) o.gpu = await run('webgpu');
          return o;
        }""", ["f16q8", spk, ins, bool(has_gpu)])
        br.close()
    srv.shutdown()
    c = np.array(res["cpu"]["pcm"], dtype=np.float32)
    check(res["cpu"]["be"] == "cpu" and len(c) > 0 and np.isfinite(c).all(), "Chromium: backend:'cpu' работает (info.backend = cpu)")
    check(res["cpu"]["ticks"] > 0, f"Chromium: страница не замирает на процессорном синтезе (таймер тикнул {res['cpu']['ticks']} раз)")
    check(len(c) == len(cpu.get("f16q8", [])) and float(np.abs(c - cpu["f16q8"]).max()) < 1e-6, "f16q8: Chromium-cpu = node-cpu (один и тот же JS)")
    if "gpu" in res:
        g = np.array(res["gpu"]["pcm"], dtype=np.float32)
        ok = res["gpu"]["be"] == "webgpu" and len(g) == len(c) and float(np.abs(g - c).max()) < 1e-3 * max(1.0, float(np.abs(g).max()))
        check(ok, f"f16q8: процессорный бэкенд = WebGPU (расхождение {float(np.abs(g - c).max()) if len(g) == len(c) else 'длина'})")
    else:
        print("SKIP: в этом Chromium нет WebGPU — сверка cpu и webgpu для q8 пропущена")
    check(not errors, "без ошибок страницы" + ("" if not errors else ": " + str(errors[:2])))
    shutil.rmtree(work, ignore_errors=True)
    print("FAILED: %d" % fails if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/vox"))
