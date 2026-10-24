"""Chatterbox in vox against the official chatterbox-tts code on a tiny model with random weights (tools/make_tiny_chatterbox.py).
Run: python tests/test_chatterbox_tiny.py [build/vox]   (needs chatterbox-tts with torch, librosa, s3tokenizer; otherwise SKIP)

What is checked:
  1. the converter: S3Gen's sizes read from the weights, the built-in voice (conds.pt) carried over unchanged;
  2. every stage (`vox verify`, f32 / bf16 / q8): text tokens, VoiceEncoder, T3 prompt tokens, T3 conditioning, the CFG
     logits of every T3 step (teacher forcing on the tokens the official T3 sampled), flow encoder and the mel after 10
     Euler steps from the same noise, HiFT F0 and the waveform after the fade-in;
  3. whole utterances through the engine (built-in voice; the official speech tokens and S3Gen's random draws injected),
     with several settings (exaggeration, cfg_weight, temperature, top_p, repetition_penalty): the same audio as
     ChatterboxTTS.generate (before its watermark) within float rounding;
  4. a voice cloned from prompt.wav against ChatterboxTTS.prepare_conditionals: S3Gen's prompt (mel, speech tokens,
     x-vector) and the VoiceEncoder embedding; T3's prompt tokens are compared and reported (the official code resamples
     to 16 kHz with soxr, vox with torchaudio's resampler — on this random tokenizer part of the tokens differ);
  5. text: punc_norm + the tokenizer against EnTokenizer on assorted texts; long text split into groups of sentences;
  6. own sampling (seeds), [happy] → exaggeration, streaming text == whole text, errors.
Not checked: the real ResembleAI/chatterbox weights (no Hugging Face access during development); GPU."""
import json
import os
import subprocess
import sys
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from test_kokoro_tiny import read_wav_f32  # noqa: E402


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def main(cli):
    try:
        import torch  # noqa: F401
        import librosa  # noqa: F401
        import chatterbox  # noqa: F401
        import s3tokenizer  # noqa: F401
        from safetensors.numpy import load_file, save_file
    except Exception as e:  # noqa: BLE001
        print("SKIP: needs chatterbox-tts (torch, librosa, s3tokenizer):", e)
        return 0
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from vox.build_chatterbox import build
    from vox.formats import read_tmod_manifest
    sys.path.remove(os.path.join(ROOT, "python"))
    for m in [m for m in sys.modules if m == "vox" or m.startswith("vox.")]:
        del sys.modules[m]

    work = tempfile.mkdtemp(prefix="voxcb_")
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_chatterbox.py"), "--out", work])
    assert rc == 0, out[-3000:]
    md = os.path.join(work, "model")
    tm = {}
    for dt, q in (("f32", ""), ("bf16", ""), ("q8", "q8")):
        tm[dt] = os.path.join(work, f"cb_{dt}.tmod")
        build(md, tm[dt], dtype="f32" if dt == "f32" else "bf16", flow_dtype="f32", sha=False, quant=q)
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures += 1

    # 1. converter
    man = read_tmod_manifest(tm["f32"])
    fl = man["cosyvoice"]["flow"]
    check(fl["encoder_blocks"] == 2 and fl["estimator_channels"] == 64 and fl["estimator_heads"] == 2 and fl["estimator_mid_blocks"] == 2
          and man["cosyvoice"]["hift"]["base_channels"] == 64 and man["chatterbox"]["t3"]["num_hidden_layers"] == 2,
          "converter: S3Gen and T3 sizes read from the weights (2 encoder blocks, estimator 64 ch × 2 heads, HiFT 64 ch, T3 2 layers)")
    # 2. stages (VOX_TEST_DEVICES="cpu;vulkan:0,strict": the flow estimator on the GPU too)
    devices = [d for d in os.environ.get("VOX_TEST_DEVICES", "cpu").split(";") if d]
    for dev in devices:
      for dt, tol in (("f32", "1e-4"), ("bf16", "3e-2"), ("q8", "3e-2")):
        rc, out = run([cli, "verify", "--model", tm[dt], "--dump", os.path.join(work, "dump.safetensors"), "--device", dev, "--tol", tol])
        check(rc == 0 and "PASS" in out, f"verify {dt} (tol {tol}) [{dev}]: text, voice encoder, T3, flow, HiFT")
        if rc != 0 or (dt == "f32" and dev == devices[0]):
            print("\n".join("      " + l for l in out.splitlines() if "max|diff|" in l or "equal" in l))

    # 3. whole utterances, built-in voice, official tokens and noise injected
    cases = json.load(open(os.path.join(work, "cases.json")))["cases"]
    vdef = os.path.join(work, "default.tvoice")
    rc, out = run([cli, "voice", "speaker", "--model", tm["f32"], "--speaker", "default", "-o", vdef])
    check(rc == 0, "built-in voice 'default' (conds.pt)" + ("" if rc == 0 else " — " + out[-300:]))

    def say(text, voice, extra, *more, model=None, dev="cpu"):
        o = os.path.join(work, "o.wav")
        rc, so = run([cli, "say", text, "--model", model or tm["f32"], "--voice", voice, "-o", o, "--float", "--device", dev,
                      "--extra", json.dumps(extra), *more])
        return rc, so, (read_wav_f32(o) if rc == 0 else None)

    for dev, name in [(d, n) for d in devices for n in cases]:
        p = os.path.join(work, f"e2e_{name}.safetensors")
        d = load_file(p)
        from safetensors import safe_open
        with safe_open(p, "np") as f:
            meta = f.metadata()
        inj = os.path.join(work, f"inj_{name}.safetensors")
        save_file({"0.tokens": d["t3_tokens"], "0.flow_noise": d["flow_noise"], "0.rand_ini": d["rand_ini"], "0.sine_noise": d["sine_noise"]}, inj)
        opts = {k: float(meta[k]) for k in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")}
        opts["noise_file"] = inj
        rc, so, got = say(meta["text"], vdef, opts, dev=dev)
        exp = d["audio"]
        ok = got is not None and len(got) == len(exp)
        err = float(np.abs(got - exp).max()) if ok else 9.9
        check(ok and err < 1e-3, f"{name} [{dev}]: engine == ChatterboxTTS.generate on the same tokens and noise ({len(exp)} samples, max |diff| {err:.1e}; "
              f"exaggeration {opts['exaggeration']}, cfg {opts['cfg_weight']})" + ("" if rc == 0 else " — " + so[-300:]))

    for dev in devices[1:]:
        rc, so = run([cli, "bench", "--model", tm["f32"], "--voice", vdef, "--text", "Short one.", "--runs", "1", "--device", dev,
                      "--extra", '{"max_new_tokens": 30}'])
        st = next((json.loads(l) for l in so.splitlines() if l.startswith("{")), {}) if rc == 0 else {}
        bk = st.get("stats", {}).get("backend", {})
        check(bk.get("flow_device") == "vulkan" and bk.get("t3_device") == "vulkan",
              f"[{dev}] the flow estimator and T3 run on the GPU (stats: flow_device={bk.get('flow_device')}, t3_device={bk.get('t3_device')}, gemm={bk.get('gemm')})")

    # 4. cloning
    prompt = os.path.join(work, "prompt.wav")
    vcl = os.path.join(work, "clone.tvoice")
    rc, out = run([cli, "voice", "create", prompt, "--model", tm["f32"], "-o", vcl])   # no --text, no --xvector-only
    check(rc == 0, "voice from prompt.wav, no transcript and no --xvector-only needed" + ("" if rc == 0 else " — " + out[-300:]))
    vcl2 = os.path.join(work, "clone2.tvoice")
    rc2, out2 = run([cli, "voice", "create", prompt, "--model", tm["f32"], "--xvector-only", "-o", vcl2])
    check(rc2 == 0 and np.array_equal(load_file(vcl2)["feat.cb_s3_tokens"], load_file(vcl)["feat.cb_s3_tokens"]),
          "--xvector-only gives the same voice")
    import vox as _vx
    _e = _vx.Engine(tm["f32"], device="cpu")
    _v = _e.clone_voice(prompt)   # Python: no ref_text either
    check(_v is not None, "Python: clone_voice(wav) without ref_text")
    del _v, _e
    v, c = load_file(vcl), load_file(os.path.join(work, "clone.safetensors"))
    s3_same = np.array_equal(v["feat.cb_s3_tokens"], c["clone_s3_tokens"])
    mel_err = float(np.abs(v["feat.cb_s3_mel"] - c["clone_s3_mel"]).max())
    xv_err = float(np.abs(v["feat.cb_xvec"] - c["clone_xvec"]).max())
    cos = float((v["feat.cb_ve"] * c["clone_ve"]).sum())
    check(s3_same and mel_err < 1e-3 and xv_err < 1e-4,
          f"clone, S3Gen prompt: {c['clone_s3_tokens'].size} speech tokens identical, mel within {mel_err:.1e}, x-vector within {xv_err:.1e}")
    check(cos > 0.999, f"clone, VoiceEncoder: cosine {cos:.6f} to the official embedding (16 kHz by torchaudio vs soxr)")
    same = float((v["feat.cb_t3_tokens"] == c["clone_t3_tokens"]).mean()) if v["feat.cb_t3_tokens"].size == c["clone_t3_tokens"].size else 0.0
    print(f"info  clone, T3 prompt tokens: {same:.0%} equal to the official ones (soxr vs torchaudio resampling of the 16 kHz copy)")
    rc, so, w = say("Hello there from a cloned voice.", vcl, {}, "--seed", "3")
    check(rc == 0 and w is not None and len(w) > 0, "a cloned voice speaks" + ("" if rc == 0 else " — " + so[-300:]))

    # 5. text
    from chatterbox.tts import punc_norm
    from chatterbox.models.tokenizers import EnTokenizer
    tok = EnTokenizer(os.path.join(md, "tokenizer.json"))
    texts = ["hello   world... it's “quoted” — fine: yes", "Already fine.", "no ending punctuation", "Wait - what; really?!",
             "Numbers 1, 2 and 3…", "élan and naïve café", "   leading spaces ", "a"]
    for t in texts:
        want = [[255] + tok.encode(punc_norm(t)) + [0]]
        rc, out = run([cli, "phonemize", "--model", tm["f32"], "--device", "cpu", t])
        got = json.loads(out)["tokens"] if rc == 0 else None
        check(got == want, f"text: {t!r} → punc_norm + tokenizer.json as EnTokenizer" + ("" if got == want else f"\n      vox:  {got}\n      want: {want}"))
    long = " ".join(f"This is sentence number {i}, which talks about the weather and other things." for i in range(1, 12))
    rc, out = run([cli, "phonemize", "--model", tm["f32"], "--device", "cpu", long])
    pieces = json.loads(out)["pieces"] if rc == 0 else []
    check(len(pieces) == 3 and all(len(p) <= 300 for p in pieces), f"long text: {len(long)} characters → {len(pieces)} groups of sentences (≤ 300)")

    # 6. sampling, markup, errors
    text = "Hello world. This is a test of the tiny model."
    a = say(text, vdef, {}, "--seed", "7")
    b = say(text, vdef, {}, "--seed", "7")
    c2 = say(text, vdef, {}, "--seed", "8")
    check(a[0] == 0 and np.array_equal(a[2], b[2]), "own T3 sampling: same seed → identical audio")
    check(c2[0] == 0 and (len(c2[2]) != len(a[2]) or not np.array_equal(c2[2], a[2])), "another seed → other audio")
    rc, out = run([cli, "say", "Hi.", "--model", tm["f32"], "--speaker", "nobody", "-o", os.path.join(work, "x.wav"), "--device", "cpu"])
    check(rc != 0 and "default" in out, "unknown speaker → clear error naming the built-in one")
    rc, so, _ = say(text, vdef, {"temperature": 0})
    check(rc != 0 and "temperature" in so, "temperature 0 → clear error")

    try:
        import vox
        vox._native()
        has_py = True
    except Exception:  # noqa: BLE001
        has_py = False
    if has_py:
        with vox.Engine(tm["f32"], device="cpu") as eng:
            caps = eng.capabilities
            check(eng.info["family"] == "chatterbox" and caps["clone_xvector"] and "default" in caps["speakers"],
                  "Python: capabilities (cloning, the built-in speaker)")
            vv = eng.load_voice(vdef)
            w1 = eng.synthesize(long, vv, seed=3)
            outp = []
            with eng.session(vv, seed=3) as s:
                for i in range(0, len(long), 41):
                    s.feed(long[i: i + 41])
                    outp.extend(s.ready())
                s.end()
                outp.extend(s)
                info = s.info
            check(np.array_equal(np.concatenate(outp), w1), f"Python: text fed in pieces == whole text ({len(info['model']['pieces'])} groups)")
            with eng.session(vv, seed=3) as s:
                s.feed("[happy] What a day!")
                s.end()
                list(s)
                info = s.info
            check(abs(info["model"]["exaggeration"] - 0.8) < 1e-6, f"[happy] → exaggeration {info['model']['exaggeration']} (manifest emotion_options)")
    else:
        print("SKIP: Python API checks (the vox package with the native module is not installed)")

    print("Chatterbox tiny:", "ALL OK" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
