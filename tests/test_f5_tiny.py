"""F5-TTS в vox против официального пакета f5_tts (+ vocos) на крошечной модели со случайными весами.
Запуск: python tests/test_f5_tiny.py [build/vox]   (нужны torch, torchaudio, f5-tts, vocos, safetensors, pypinyin)

Что проверяется:
  1. каждая стадия по отдельности (токенизация, mel образца, текстовый энкодер, DiT с обеими ветками CFG, Vocos, цикл CFM
     с EPSS и sway) — `vox verify` на f32-модели, на CPU и (если есть) на Vulkan;
  2. сквозной звук (нарезка текста, формула длины куска, длительность, кроссфейд, подъём громкости образца) против настоящего
     infer_batch_process в четырёх сценариях, шум подставлен одинаковый;
  3. bf16 и q8 — в пределах допуска квантования;
  4. детерминизм по seed, независимость от способа подачи текста, голос .tvoice, понятные ошибки.
Что НЕ проверяется: качество на настоящих весах (их здесь нет) и скорость на реальном GPU."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
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


def load_st(path):
    from safetensors.numpy import load_file
    return load_file(path)


def main(cli):
    try:
        import torch  # noqa: F401
        import f5_tts  # noqa: F401
        import vocos  # noqa: F401
        from safetensors.numpy import save_file  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("SKIP: needs torch, f5-tts, vocos, safetensors:", e)
        return 0
    try:
        from vox.build_f5 import build   # установленный пакет (pip install .), с нативным модулем
    except ImportError:
        sys.path[:0] = [os.path.join(ROOT, "python")]
        from vox.build_f5 import build   # только чистый Python: проверки Python API будут пропущены

    work = tempfile.mkdtemp(prefix="voxf5_")
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_f5.py"), "--out", work])
    assert rc == 0, out[-2000:]
    ck, vc = os.path.join(work, "ckpt", "model.safetensors"), os.path.join(work, "vocos")
    vocab = os.path.join(work, "ckpt", "vocab.txt")
    tm = {}
    for name, kw in (("f32", dict(dtype="f32")), ("bf16", dict(dtype="bf16")), ("q8", dict(dtype="f32", quant="q8"))):
        tm[name] = os.path.join(work, f"tiny_{name}.tmod")
        build(ck, vocab, vc, tm[name], dim_head=32, **kw)
    e2e = json.load(open(os.path.join(work, "e2e.json")))
    ref_wav = os.path.join(work, "ref.wav")
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures += 1

    devices = [d for d in os.environ.get("VOX_TEST_DEVICES", "cpu").split(";") if d]

    # 1. стадии
    for dev in devices:
        gpu = not dev.startswith("cpu")
        for dump in ("dump.safetensors", "dump_long.safetensors"):
            rc, out = run([cli, "verify", "--model", tm["f32"], "--dump", os.path.join(work, dump), "--device", dev, "--tol", "1e-4" if not gpu else "1e-3"])
            check(rc == 0 and "PASS" in out, f"verify f32 {dump} [{dev}]")
            if rc != 0:
                print(out)
    # 3. квантованные модели — допуск квантования
    for name in ("bf16", "q8"):
        for dump in ("dump.safetensors", "dump_long.safetensors"):
            rc, out = run([cli, "verify", "--model", tm[name], "--dump", os.path.join(work, dump), "--device", "cpu", "--tol", "1e-2"])
            check(rc == 0 and "PASS" in out, f"verify {name} {dump} (tol 1e-2)")
            if rc != 0:
                print(out)

    # 2. сквозной звук
    voice = os.path.join(work, "v.tvoice")
    rc, out = run([cli, "voice", "create", ref_wav, "--model", tm["f32"], "--text", e2e["ref_text"], "-o", voice, "--device", "cpu"])
    assert rc == 0, out
    for sc, p in e2e["scenarios"].items():
        exp = load_st(os.path.join(work, f"expected_{sc}.safetensors"))["wave"]
        extra = {"noise_file": os.path.join(work, f"noise_{sc}.safetensors")}
        if p["max_chars"]:
            extra["max_chars"] = p["max_chars"]
        if p["cross_fade"] != 0.15:
            extra["cross_fade_duration"] = p["cross_fade"]
        for dev in devices[:1]:
            outp = os.path.join(work, f"out_{sc}.wav")
            rc, so = run([cli, "say", p["gen_text"], "--model", tm["f32"], "--voice", voice, "-o", outp, "--float", "--device", dev,
                          "--nfe", str(p["nfe"]), "--speed", str(p["speed"]), "--extra", json.dumps(extra)])
            assert rc == 0, so
            got = read_wav_f32(outp)
            same_len = len(got) == len(exp)
            d = float(np.abs(got[:min(len(got), len(exp))] - exp[:min(len(got), len(exp))]).max()) if len(got) and len(exp) else 1.0
            check(same_len and d < 1e-6, f"end-to-end {sc}: {p['chunks']} chunk(s), {len(exp)} samples, max|diff| {d:.2e} (ref max {np.abs(exp).max():.2e})")
    # квантованные: коррелируют с эталоном
    sc = "chunks"
    p = e2e["scenarios"][sc]
    exp = load_st(os.path.join(work, f"expected_{sc}.safetensors"))["wave"]
    for name in ("bf16", "q8"):
        outp = os.path.join(work, f"out_{name}.wav")
        v2 = os.path.join(work, f"v_{name}.tvoice")
        assert run([cli, "voice", "create", ref_wav, "--model", tm[name], "--text", e2e["ref_text"], "-o", v2, "--device", "cpu"])[0] == 0
        rc, so = run([cli, "say", p["gen_text"], "--model", tm[name], "--voice", v2, "-o", outp, "--float", "--device", "cpu", "--nfe", str(p["nfe"]),
                      "--extra", json.dumps({"noise_file": os.path.join(work, f"noise_{sc}.safetensors"), "max_chars": p["max_chars"]})])
        assert rc == 0, so
        got = read_wav_f32(outp)
        rel = float(np.linalg.norm(got - exp) / np.linalg.norm(exp)) if len(got) == len(exp) else 9.9
        check(rel < 0.05, f"end-to-end {name}: relative L2 error {rel:.3e} (< 5e-2)")

    # 4. детерминизм, подача текста частями, голос, ошибки
    text = "Seed test. Another sentence, with commas; and more!"
    def say(seed, model=tm["f32"], extra=None, text=text):
        o = os.path.join(work, f"s{seed}.wav")
        cmd = [cli, "say", text, "--model", model, "--voice", voice, "-o", o, "--float", "--device", "cpu", "--seed", str(seed), "--nfe", "4"]
        if extra:
            cmd += ["--extra", json.dumps(extra)]
        rc, so = run(cmd)
        return rc, so, (read_wav_f32(o) if rc == 0 else None)
    r1, r2, r3 = say(5), say(5), say(6)
    check(r1[0] == 0 and np.array_equal(r1[2], r2[2]), "same seed → identical audio")
    check(r3[0] == 0 and not np.array_equal(r1[2], r3[2]), "different seed → different audio")
    check(np.isfinite(r1[2]).all() and np.abs(r1[2]).max() < 1.0, "audio is finite and within [-1, 1]")
    try:
        import vox
        vox._native()
        has_py = True
    except Exception:  # noqa: BLE001
        has_py = False
    if has_py:
        with vox.Engine(tm["f32"], device="cpu") as eng:
            check(eng.info["family"] == "f5-tts" and eng.capabilities["clone_icl"], "Python: engine info / capabilities")
            v = eng.clone_voice(ref_wav, e2e["ref_text"], name="tiny", seed=5)
            whole = eng.synthesize(text, voice=v, seed=5, nfe_step=4)
            # библиотека и CLI могут собираться с разными флагами (FMA и т.п.): внутри одной сборки результат бит-в-бит, между сборками — до ~1e-8
            check(len(whole) == len(r1[2]) and float(np.abs(whole - r1[2]).max()) < 1e-6, "Python library ≈ CLI (same seed)")
            out = []
            with eng.session(v, seed=5, nfe_step=4) as s:
                for piece in ["Seed te", "st. Another sent", "ence, with com", "mas; and mo", "re!"]:
                    s.feed(piece)
                    out.extend(s.ready())
                s.end()
                out.extend(s)
            check(np.array_equal(np.concatenate(out), whole), "text fed in pieces == whole text")

            # --- двусторонний стриминг (F5 озвучивает куски целиком: text_in = "chunk") ---
            caps = eng.capabilities["stream"]
            check(caps["text_in"] == "chunk" and caps["audio_out"] == "chunks" and caps["exact"], f"capabilities.stream у F5: {caps}")
            sp = dict(seed=5, nfe_step=4)
            # auto: звук идёт до конца текста (готовые куски), а итог = синтез целиком (exact)
            with eng.duplex(v, **sp) as dx:
                dx.write("Seed test. Another sentence, ")
                dx.write("with commas; and more!")
                time.sleep(0.05)
                dx.end()
                got = np.concatenate(list(dx.audio()))
            check(np.array_equal(got, whole), "duplex auto: текст порциями → тот же звук, что у синтеза целиком")
            # manual: до end() ничего не озвучивается
            with eng.duplex(v, commit="manual", **sp) as dx:
                for piece in ["Seed te", "st. Another sent", "ence, with com", "mas; and mo", "re!"]:
                    dx.write(piece)
                check(dx.read(timeout=0.3) is None and dx.info()["committed_bytes"] == 0, "duplex manual: пока текст не закончен, звука нет")
                got = dx.final(timeout=120)
            check(np.array_equal(got, whole), "duplex manual: набор → одна запись = синтез целиком")
            # sentence: законченное предложение озвучивается сразу, не дожидаясь следующего
            with eng.duplex(v, commit="sentence", **sp) as dx:
                dx.write("Seed test. Another sen")
                first = dx.read(timeout=120)
                inf = dx.info()
                check(first is not None and len(first) > 0 and inf["pending"] == "Another sen" and not dx.done,
                      f"duplex sentence: первое предложение озвучено, пока второе набирается ({0 if first is None else len(first)} отсч.)")
                dx.write("tence, with commas; and more!")
                rest = dx.final(timeout=120)
                full = np.concatenate([first, rest])
            check(len(full) > len(first) and np.isfinite(full).all() and np.abs(full).max() < 1.0,
                  "duplex sentence: итоговый звук конечен и в пределах [-1, 1] (куски другие, чем у целого текста — это ожидаемо)")
            # flush: незаконченное предложение озвучивается по требованию, хвост отдаётся с затуханием
            with eng.duplex(v, commit="manual", **sp) as dx:
                dx.write("Seed test")
                dx.flush()
                parts = [dx.read(timeout=120)]
                while parts[-1] is not None:
                    nxt = dx.read(timeout=1.0)   # ждём, пока фраза дочитается до конца
                    if nxt is None:
                        break
                    parts.append(nxt)
                first = np.concatenate(parts) if parts[0] is not None else None
                check(first is not None and len(first) > 1000 and not dx.done and abs(float(first[-1])) < 1e-6,
                      f"duplex flush: фраза озвучена целиком без end(), конец затухает до нуля (|last|={0 if first is None else abs(float(first[-1])):.2e})")
                dx.write(" again.")
                tail = dx.final(timeout=120)
                check(len(tail) > 0 and np.isfinite(tail).all(), "duplex flush: после flush продолжение озвучивается новым куском")
            # idle: набор замолчал — озвучивается накопленное
            with eng.duplex(v, commit="manual", idle_ms=80, **sp) as dx:
                dx.write("Idle phrase")
                first = dx.read(timeout=120)
                check(first is not None and len(first) > 0 and dx.info()["idle_flushes"] == 1, "duplex idle_ms: накопленное озвучено после паузы в наборе")
                dx.cancel()
            vp = os.path.join(work, "saved.tvoice")
            v.save(vp)
            again = eng.synthesize(text, voice=vp, seed=5, nfe_step=4)
            check(np.array_equal(again, whole), ".tvoice round trip keeps the voice")
            try:
                eng.synthesize(text, voice=v, seed=5, fix_duration=3.0)
                check(False, "fix_duration is rejected")
            except Exception as ex:  # noqa: BLE001
                check("fix_duration" in str(ex), "fix_duration is rejected with a clear message")
            try:
                eng.synthesize(text, voice=v, seed=5, speed=0.0)
                check(False, "speed=0 is rejected")
            except Exception as ex:  # noqa: BLE001
                check("speed" in str(ex), "speed=0 is rejected")
            st = eng.stats()
            check(st.get("backend", {}).get("dit_ms", 0) > 0, "stats report DiT time")
    else:
        print("SKIP: Python API checks (the vox package with the native module is not installed)")

    # голос, сделанный для другой модели, пересобирается из записи образца, которая лежит внутри .tvoice
    tmodel = os.path.join(work, "test.tmod")
    assert run([cli, "make-test-model", "-o", tmodel])[0] == 0
    tv = os.path.join(work, "test.tvoice")
    assert run([cli, "voice", "create", ref_wav, "--model", tmodel, "--text", "x", "-o", tv])[0] == 0
    ob = os.path.join(work, "bad.wav")
    rc, so = run([cli, "say", "hi", "--model", tm["f32"], "--voice", tv, "-o", ob, "--float", "--nfe", "4"])
    check(rc == 0 and os.path.getsize(ob) > 1000, "a voice made for another model is rebuilt from its stored reference audio")
    # слишком короткий образец — понятная ошибка
    short = os.path.join(work, "short.wav")
    import wave
    with wave.open(short, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x00\x01" * 200)
    rc, so = run([cli, "voice", "create", short, "--model", tm["f32"], "--text", "x", "-o", os.path.join(work, "short.tvoice")])
    check(rc != 0 and "too short" in so, "too short reference audio → clear error")

    print("F5 tiny: " + ("ALL OK" if failures == 0 else f"{failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
