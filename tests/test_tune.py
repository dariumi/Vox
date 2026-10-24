"""vox tune: candidates, measurement in child processes, the cache, and engines that open with the measured configuration.
Run: python tests/test_tune.py [build/vox]  (the vox-test model; with a Vulkan device, VOX_TUNE_SOFTWARE=1 lets lavapipe in)"""
import json
import os
import subprocess
import sys
import tempfile
import wave

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(cli):
    work = tempfile.mkdtemp(prefix="voxtune_")
    env = dict(os.environ, VOX_TUNE_CACHE=os.path.join(work, "tune.json"), VOX_TUNE_SOFTWARE="1")
    run = lambda *a, **kw: subprocess.run([cli, *a], capture_output=True, text=True, env=env, **kw)
    model, voice, ref = (os.path.join(work, x) for x in ("m.tmod", "v.tvoice", "ref.wav"))
    assert run("make-test-model", "-o", model).returncode == 0
    with wave.open(ref, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes((np.sin(2 * np.pi * 220 * np.arange(24000) / 24000) * 8000).astype("<i2").tobytes())
    assert run("voice", "create", ref, "--model", model, "--text", "hello", "-o", voice).returncode == 0
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        fails += not ok

    info = json.loads(run("tune", "--model", model, "--show").stdout)
    devs = [c["device"] for c in info["candidates"]]
    check("cpu" in devs and info["entry"] is None, f"candidates before tuning: {devs}; nothing cached yet")
    r = run("tune", "--model", model, "--voice", voice, "--runs", "1", "--text", "Привет, мир. Это проверка.")
    entry = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    ok_rows = [c for c in entry.get("candidates", []) if "rtf" in c]
    check(r.returncode == 0 and entry.get("device") in devs and len(ok_rows) == len(devs),
          f"vox tune measured {len(ok_rows)}/{len(devs)} configurations, best {entry.get('device')} (RTF {entry.get('rtf', 0):.3f})"
          + ("" if r.returncode == 0 else " — " + r.stderr[-300:]))
    best = min(ok_rows, key=lambda c: c["rtf"]) if ok_rows else {}
    check(entry.get("device") == best.get("device"), "the saved one is the lowest RTF")
    info2 = json.loads(run("tune", "--model", model, "--show").stdout)
    check(info2["entry"] and info2["entry"]["device"] == entry["device"] and info2["entry"]["hardware_key"] == info2["hardware_key"],
          "the cache holds it under this model and this hardware")
    plan = run("plan", "--model", model).stdout + run("plan", "--model", model).stderr
    check("tuned: " + entry["device"] in plan, "vox plan (device auto) uses it")
    plan_off = subprocess.run([cli, "plan", "--model", model], capture_output=True, text=True, env=dict(env, VOX_TUNE="0"))
    check("tuned:" not in plan_off.stdout + plan_off.stderr, "VOX_TUNE=0: ignored")
    plan_cpu = run("plan", "--model", model, "--device", "cpu")
    check("tuned:" not in plan_cpu.stdout + plan_cpu.stderr, "an explicit --device wins over the cache")
    # Python: info, the engine opening with it, vox.tune()
    py = subprocess.run([sys.executable, "-c", f"""
import json, vox
e = vox.Engine({model!r})
print(json.dumps({{"tuned": e.info.get("tuned"), "plain": vox.Engine({model!r}, resources={{"tuned": False}}).info.get("tuned")}}))
"""], capture_output=True, text=True, env=env)
    pj = json.loads(py.stdout.strip().splitlines()[-1]) if py.returncode == 0 else {}
    check(pj.get("tuned") and pj["tuned"]["device"] == entry["device"] and pj.get("plain") is None,
          "Python: Engine(model) opens with the tuned configuration, resources={'tuned': False} without" + ("" if py.returncode == 0 else " — " + py.stderr[-300:]))
    py2 = subprocess.run([sys.executable, "-c", f"""
import json, vox
rows = []
e = vox.tune({model!r}, {voice!r}, runs=1, text="Привет, мир.", candidates=[{{"label": "cpu", "device": "cpu"}}], progress=rows.append)
print(json.dumps({{"entry": e, "rows": rows}}))
"""], capture_output=True, text=True, env=env)
    p2 = json.loads(py2.stdout.strip().splitlines()[-1]) if py2.returncode == 0 else {}
    check(p2.get("entry", {}).get("device") == "cpu" and len(p2.get("rows", [])) == 1,
          "Python: vox.tune() measures in a child process, reports progress and saves" + ("" if py2.returncode == 0 else " — " + py2.stderr[-300:]))
    r = run("tune", "--model", model, "--forget")
    info3 = json.loads(run("tune", "--model", model, "--show").stdout)
    check(r.returncode == 0 and info3["entry"] is None, "vox tune --forget")
    print("tune: ALL OK" if not fails else f"tune: {fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
