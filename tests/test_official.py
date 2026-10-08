#!/usr/bin/env python3
"""vox against the OFFICIAL qwen_tts implementation (PyTorch) on a tiny random-weight model.

What is checked (all with the reference code, not a re-implementation):
  1. native voice creation (`vox voice create`): speaker embedding (mel + ECAPA-TDNN) and reference codes
     (Mimi encoder + split RVQ) vs `Qwen3TTSModel.create_voice_clone_prompt`;
  2. talker / code-predictor logits on every step and the codec decoder wav vs `generate_voice_clone` (greedy),
     teacher-forced and free-running (`vox verify`), for ICL and x-vector voices;
  3. decoder chunking (qwen_tts chunked_decode, 300 frames + 25 context): a reference that pushes the speech across
     the chunk boundary, and a reference that is itself longer than one chunk;
  4. the same on every weight format (f32 / f16 / bf16 / q8 with distribution metrics) and on every device given
     in VOX_TEST_DEVICES (default "cpu"; e.g. "cpu;vulkan:0").

Needs: torch, qwen-tts==0.1.1, transformers==4.57.3, librosa, a built ./build/vox and a Qwen2 vocab gguf
(VOX_QWEN2_GGUF; without it the test is skipped).

    VOX_QWEN2_GGUF=qwen2.gguf VOX_TEST_DEVICES="cpu;vulkan:0" python tests/test_official.py build/vox
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
try:  # prefer the installed package (it has the native module); fall back to the sources
    import vox  # noqa: F401
    if getattr(vox, "_vox", None) is None:
        raise ImportError
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "python"))
os.environ.setdefault("PYTHONPATH", os.path.join(ROOT, "python"))

GGUF = os.environ.get("VOX_QWEN2_GGUF", "")
DEVICES = [d for d in os.environ.get("VOX_TEST_DEVICES", "cpu").split(";") if d]


def run(cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python")))
    if check and r.returncode != 0:
        print("$", " ".join(map(str, cmd)))
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"command failed ({r.returncode})")
    return r


def synth_ref(path, seconds, seed):
    from vox.audio import save_wav
    sr = 24000
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    f0 = 150 + 60 * np.sin(2 * np.pi * 0.3 * t)
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(f0) / sr) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.1 * t)) + 0.03 * rng.normal(size=t.size)
    save_wav(path, x.astype(np.float32), sr)


def main(cli):
    if not GGUF or not os.path.exists(GGUF):
        print("SKIP: set VOX_QWEN2_GGUF=path/to/ggml-vocab-qwen2.gguf")
        return 0
    try:
        import torch  # noqa: F401
        import qwen_tts  # noqa: F401
    except ImportError as e:
        print(f"SKIP: reference implementation not installed ({e}); pip install torch qwen-tts==0.1.1")
        return 0
    from vox.formats import VoiceProfile

    work = tempfile.mkdtemp(prefix="vox_official_")
    model = os.path.join(work, "model")
    run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_official.py"), "--vocab", GGUF, "--out", model])
    tmods = {}
    for name, extra in (("f32", ["--dtype", "f32"]), ("f16", ["--dtype", "f16"]), ("bf16", ["--dtype", "bf16"]),
                        ("q8", ["--dtype", "f32", "--quant", "q8"]), ("bf16c", ["--dtype", "bf16", "--codec-dtype", "bf16"])):
        tmods[name] = os.path.join(work, f"tiny_{name}.tmod")
        run([sys.executable, "-m", "vox.build", "--model", model, "-o", tmods[name]] + extra)

    failures = 0

    def check(cond, msg):
        nonlocal failures
        print(("PASS " if cond else "FAIL ") + msg)
        failures += 0 if cond else 1

    # ---- 1. voices: official export vs native creation ----
    refs = {"short": (3.3, 1), "long": (21.0, 2), "huge": (25.5, 3)}
    voices = {}
    for key, (secs, seed) in refs.items():
        wav = os.path.join(work, f"ref_{key}.wav")
        synth_ref(wav, secs, seed)
        off = os.path.join(work, f"off_{key}.tvoice")
        nat = os.path.join(work, f"nat_{key}.tvoice")
        text = "Привет, это референс голоса для проверки."
        run([sys.executable, os.path.join(ROOT, "tools", "export_voice.py"), "--model", model, "--ref", wav, "--text", text,
             "--lang", "english", "-o", off])
        run([cli, "voice", "create", wav, "--model", tmods["f32"], "--text", text, "--lang", "english", "-o", nat])
        a, b = VoiceProfile.load(off), VoiceProfile.load(nat)
        emb_err = float(np.abs(a.speaker_emb - b.speaker_emb).max() / max(1.0, np.abs(a.speaker_emb).max()))
        same_shape = a.ref_codes.shape == b.ref_codes.shape
        code_eq = float((a.ref_codes == b.ref_codes).mean()) if same_shape else 0.0
        check(emb_err < 1e-4, f"voice[{key}] speaker embedding vs official: rel err {emb_err:.2e}")
        check(same_shape and code_eq == 1.0, f"voice[{key}] reference codes vs official: {a.ref_codes.shape} vs {b.ref_codes.shape}, "
                                             f"identical {code_eq * 100:.2f}%")
        check(np.array_equal(a.ref_text_tokens, b.ref_text_tokens), f"voice[{key}] reference text tokens")
        voices[key] = off
    # Python API (Engine.clone_voice) must produce the same profile as the CLI
    try:
        import vox as vx
        if vx._vox is not None:
            eng = vx.Engine(tmods["f32"])
            pv = eng.clone_voice(os.path.join(work, "ref_short.wav"), "Привет, это референс голоса для проверки.")
            pyp = os.path.join(work, "py_short.tvoice")
            pv.save(pyp)
            a, b = VoiceProfile.load(os.path.join(work, "nat_short.tvoice")), VoiceProfile.load(pyp)
            # the CLI and the Python module may be built with different flags (-march=native → FMA): compare with tolerance
            err = float(np.abs(a.speaker_emb - b.speaker_emb).max() / max(1.0, np.abs(a.speaker_emb).max()))
            check(err < 1e-5 and np.array_equal(a.ref_codes, b.ref_codes),
                  f"Python Engine.clone_voice == vox voice create (emb rel err {err:.1e}, codes identical)")
    except ImportError:
        print("note: native Python module not installed; Engine.clone_voice not checked")
    xv = os.path.join(work, "off_xv.tvoice")
    run([sys.executable, os.path.join(ROOT, "tools", "export_voice.py"), "--model", model, "--ref",
         os.path.join(work, "ref_short.wav"), "--xvector-only", "--lang", "english", "-o", xv])
    voices["xv"] = xv

    # ---- 2/3. reference dumps (official generate, greedy) ----
    cases = [("short", "Hello, this is a test.", 30), ("short", "Привет! Это проверка, слово за словом.", 30),
             ("xv", "Hello world, x-vector voice.", 30), ("long", "Crossing the decoder chunk boundary.", 80),
             ("huge", "Reference longer than one decoder chunk.", 120)]
    dumps = []
    for i, (vk, text, frames) in enumerate(cases):
        d = os.path.join(work, f"dump{i}")
        phr = os.path.join(work, f"phrase{i}.txt")
        open(phr, "w", encoding="utf-8").write(text + "\n")
        run([sys.executable, os.path.join(ROOT, "tools", "dump_reference.py"), "--model", model, "--voice", voices[vk],
             "--lang", "english", "--seeds", "1", "--max-frames", str(frames), "--phrases", phr, "--out", d])
        dumps.append((vk, os.path.join(d, "case_00_seed1.safetensors")))

    # ---- 4. vox verify on every format and device ----
    tol = {"f32": "2e-3", "f16": "2e-2", "bf16": "5e-2", "bf16c": "1e-1"}  # bf16c: bf16 weights in the codec decoder too
    for dev in DEVICES:
        for fmt, tm in tmods.items():
            for i, (vk, dump) in enumerate(dumps):
                if fmt != "f32" and i not in (0, 2, 3):
                    continue
                cmd = [cli, "verify", "--device", dev, "--model", tm, "--dump", dump, "--voice", voices[vk]]
                if fmt == "q8":
                    cmd += ["--quant", "1"]
                else:
                    cmd += ["--tol", tol[fmt]]
                    if fmt != "f32":
                        cmd += ["--no-argmax", "--free", "0"]
                    if fmt == "bf16c":  # rounded decoder weights: audio within 1 (tol*10), logits as bf16
                        cmd[cmd.index("--tol") + 1] = "5e-2"
                        cmd += ["--wav-tol", "1.0"]
                r = run(cmd, check=False)
                ok = r.returncode == 0
                check(ok, f"verify {dev:>10} {fmt:>4} case{i} [{vk}]")
                if not ok:
                    print(r.stdout[-3000:], r.stderr[-2000:])

    # ---- 5. predictor loop on the GPU (sampling in a shader) == the same decode with CPU sampling ----
    for dev in DEVICES:
        if not dev.startswith("vulkan"):
            continue
        outs = []
        for gs in ("1", "0"):
            o = os.path.join(work, f"gs{gs}_{len(outs)}.wav")
            r = subprocess.run([str(cli), "say", "Hello, this is a sampling test for the code predictor.", "--model", tmods["f32"],
                                "--voice", voices["short"], "--device", dev, "--seed", "11", "--max-frames", "60", "--temperature", "0.9",
                                "--sub-temperature", "0.9", "-o", o], capture_output=True, text=True,
                               env=dict(os.environ, VOX_GPU_SAMPLING=gs, VOX_DEBUG_NO_EOS="1"))
            outs.append(open(o, "rb").read() if r.returncode == 0 and os.path.exists(o) else b"")
        check(outs[0] and outs[0] == outs[1], f"GPU sampling == CPU sampling ({dev}, seed 11, 60 frames, T=0.9)")

    # ---- 5b. лимит памяти: сессия, не влезающая в остаток лимита, не создаётся; освобождённая сессия возвращает место ----
    try:
        import vox as vx
        if vx._vox is not None:
            with vx.Engine(tmods["f32"], resources={"device": "cpu"}) as e0:
                v0 = e0.load_voice(voices["short"])
                s0 = e0.session(v0, seed=1, max_frames=200)
                cost = e0.stats()["memory"]["sessions_reserved"]
                static = e0.plan["ram_static"]
                s0.cancel()
            check(cost > 0 and static > 0, f"session cost is accounted (reserved {cost / 2**20:.1f} MiB, static {static / 2**20:.1f} MiB)")
            limit = int(static + cost * 1.5)
            with vx.Engine(tmods["f32"], resources={"device": "cpu", "ram_limit": limit}) as e1:
                v1 = e1.load_voice(voices["short"])
                a = e1.session(v1, seed=1, max_frames=200)
                try:
                    e1.session(v1, seed=2, max_frames=200)
                    check(False, "second session must not fit into the RAM limit")
                except vx.OutOfMemory as ex:
                    check("limit" in str(ex).lower() or "memory" in str(ex).lower(), "second session refused by the RAM limit: " + str(ex)[:90])
                del a
                b = e1.session(v1, seed=3, max_frames=200)
                check(b is not None, "released session frees its share of the limit")
            try:
                vx.Engine(tmods["f32"], resources={"device": "cpu", "ram_limit": int(static * 0.5)})
                check(False, "ram_limit below the static footprint must refuse to open")
            except vx.OutOfMemory:
                check(True, "ram_limit below the weights footprint refuses to open (OutOfMemory)")
    except ImportError:
        pass


    # ---- 6. CustomVoice (готовые дикторы, диалекты, инструкция) и VoiceDesign против official generate_custom_voice /
    #         generate_voice_design (потоковый режим текста, non_streaming_mode=False) ----
    named_cases = {"custom_voice": [("english", "dylan", ""), ("english", "aiden", "Speak slowly and calmly"),
                                    ("chinese", "dylan", ""), ("auto", "eric", "Cheerful tone")],
                   "voice_design": [("english", "", "A deep calm male voice")]}
    phrase = os.path.join(work, "named_phrase.txt")
    open(phrase, "w", encoding="utf-8").write("Hello, this is a speaker test.\n")
    for variant, vcases in named_cases.items():
        vdir = os.path.join(work, variant)
        run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_official.py"), "--vocab", GGUF, "--out", vdir, "--variant", variant])
        for fmt, extra in (("f32", ["--dtype", "f32"]), ("q8", ["--dtype", "f32", "--quant", "q8"])):
            vt = os.path.join(work, f"{variant}_{fmt}.tmod")
            run([sys.executable, "-m", "vox.build", "--model", vdir, "-o", vt] + extra)
            for ci, (lang, spk, ins) in enumerate(vcases):
                if fmt == "q8" and ci:
                    continue
                dd = os.path.join(work, f"dump_{variant}_{ci}")
                if fmt == "f32":
                    run([sys.executable, os.path.join(ROOT, "tools", "dump_reference.py"), "--model", vdir, "--lang", lang, "--seeds", "1",
                         "--max-frames", "30", "--phrases", phrase, "--out", dd] + (["--speaker", spk] if spk else []) +
                        (["--instruct", ins] if ins else []))
                for dev in DEVICES:
                    cmd = [cli, "verify", "--device", dev, "--model", vt, "--dump", os.path.join(dd, "case_00_seed1.safetensors")]
                    cmd += ["--quant", "1"] if fmt == "q8" else ["--tol", "2e-3"]
                    r = run(cmd, check=False)
                    check(r.returncode == 0, f"verify {dev:>10} {fmt:>4} {variant} [{lang}/{spk or '-'}/{ins or '-'}]")
                    if r.returncode:
                        print(r.stdout[-2000:], r.stderr[-1000:])
        # Python API: capabilities, speaker_voice / design_voice, ошибки
        try:
            import vox as vx
            if vx._vox is not None:
                with vx.Engine(os.path.join(work, f"{variant}_f32.tmod")) as eng:
                    caps = eng.capabilities
                    if variant == "custom_voice":
                        check(caps["named_speakers"] and "dylan" in eng.speakers and caps["instruct"], "custom_voice capabilities")
                        v = eng.speaker_voice("Aiden", instruct="Speak slowly")
                        a1 = eng.synthesize("Hello there.", v, seed=3, max_frames=20)
                        a2 = eng.synthesize("Hello there.", eng.speaker_voice("aiden", instruct="Speak slowly"), seed=3, max_frames=20)
                        check(a1.size > 0 and np.array_equal(a1, a2), "speaker_voice is deterministic (case-insensitive name)")
                        vp = os.path.join(work, "named.tvoice")
                        v.save(vp)
                        a3 = eng.synthesize("Hello there.", vp, seed=3, max_frames=20)
                        check(np.array_equal(a1, a3), ".tvoice round trip of a speaker voice")
                        for bad, why in ((dict(speaker="nobody"), "unknown speaker"),):
                            try:
                                eng.speaker_voice(**bad)
                                check(False, why)
                            except vx.VoxError as e:
                                check("nobody" in str(e), f"{why} is refused with a clear message")
                        try:
                            eng.clone_voice((np.zeros(24000, np.float32), 24000), ref_text="x")
                            check(False, "clone on CustomVoice must be refused")
                        except vx.VoxError as e:
                            check(True, "cloning on a model without speaker encoder is refused: " + str(e)[:60])
                    else:
                        check(caps["requires_instruct"] and not caps["named_speakers"], "voice_design capabilities")
                        v = eng.design_voice("A deep calm male voice")
                        check(eng.synthesize("Hello there.", v, seed=1, max_frames=20).size > 0, "design_voice synthesizes")
                        try:
                            eng.speaker_voice(speaker="x")
                            check(False, "speaker on VoiceDesign must be refused")
                        except vx.VoxError:
                            check(True, "VoiceDesign without instruction / with speaker is refused")
        except ImportError:
            pass

    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")) else 0)
