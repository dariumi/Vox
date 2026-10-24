"""CosyVoice2 in vox against the official CosyVoice code on a tiny model with random weights (tools/make_tiny_cosyvoice.py).
Run: VOX_COSYVOICE_SRC=CosyVoice VOX_QWEN2_GGUF=qwen2.gguf python tests/test_cosyvoice_tiny.py [build/vox]
(needs a clone of https://github.com/FunAudioLLM/CosyVoice with third_party/Matcha-TTS, torch, torchaudio, onnx,
onnxruntime, s3tokenizer, transformers, inflect, pyyaml; otherwise SKIP)

What is checked:
  1. the converter: CAM++ walked out of campplus.onnx reproduces onnxruntime; the flow noise equals the one
     CausalConditionalCFM draws; CosyVoice2's special tokens get the ids the official tokenizer gives them;
  2. every stage (`vox verify`, three modes — zero-shot, cross-lingual, instruct2): text tokens, 24 kHz mel, Kaldi fbank,
     CAM++ x-vector, Whisper log-mel, S3 speech tokens, LLM log-probabilities of every step (teacher forcing on the tokens
     the official LLM sampled), flow encoder and the mel after 10 Euler steps, HiFT F0 and waveform — f32 tight,
     bf16 within quantization;
  3. whole utterances through the engine (a voice made from prompt.wav, the official LLM's tokens and HiFT noise
     injected): the same audio as CosyVoice2Model.token2wav, sample for sample within float rounding;
  4. the text frontend against CosyVoiceFrontEnd.text_normalize (numbers, grouping of sentences, Chinese rules);
  5. own sampling (seeds), speed, streaming text == whole text, markup ([laugh] → [laughter], [happy] → instruct2),
     profiles (style with an instruction), errors.
Not checked: the real CosyVoice2-0.5B weights (no Hugging Face access during development); GPU (the model runs on the CPU)."""
import json
import os
import subprocess
import sys
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from test_kokoro_tiny import read_wav_f32, mel_distance  # noqa: E402


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def main(cli):
    src = os.environ.get("VOX_COSYVOICE_SRC", "")
    gguf = os.environ.get("VOX_QWEN2_GGUF", "qwen2.gguf")
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import s3tokenizer  # noqa: F401
        import inflect  # noqa: F401
        from safetensors.numpy import load_file, save_file
    except Exception as e:  # noqa: BLE001
        print("SKIP: needs torch, torchaudio, onnx, onnxruntime, s3tokenizer, inflect:", e)
        return 0
    if not src or not os.path.isdir(os.path.join(src, "cosyvoice")):
        print("SKIP: set VOX_COSYVOICE_SRC to a clone of https://github.com/FunAudioLLM/CosyVoice (with third_party/Matcha-TTS)")
        return 0
    if not os.path.exists(gguf):
        print("SKIP: set VOX_QWEN2_GGUF=path/to/ggml-vocab-qwen2.gguf")
        return 0
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from vox.build_cosyvoice import build, campplus_from_onnx, campplus_numpy
    from vox.formats import read_tmod_manifest
    sys.path.remove(os.path.join(ROOT, "python"))
    for m in [m for m in sys.modules if m == "vox" or m.startswith("vox.")]:
        del sys.modules[m]

    work = tempfile.mkdtemp(prefix="voxcosy_")
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_cosyvoice.py"), "--vocab", gguf, "--out", work, "--src", src])
    assert rc == 0, out[-3000:]
    md = os.path.join(work, "model")
    tm = {}
    for dt in ("f32", "bf16"):
        tm[dt] = os.path.join(work, f"cosy_{dt}.tmod")
        # the official reference below runs campplus.onnx in the installed onnxruntime: follow it (the default, "window",
        # is what onnxruntime ≤ 1.25 computes — the official CosyVoice pins 1.18)
        build(md, tm[dt], dtype=dt, flow_dtype="f32", sha=False, campplus_pool="auto")
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures += 1

    # 1. converter
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import cosyvoice_ref
    cosyvoice_ref.setup(src)
    from cosyvoice.tokenizer.tokenizer import CosyVoice2Tokenizer
    tok = CosyVoice2Tokenizer(os.path.join(md, "CosyVoice-BlankEN"))
    ids = read_tmod_manifest(tm["f32"]).get("tokenizer", {}).get("cosyvoice_specials", {})
    want = {s: tok.tokenizer.convert_tokens_to_ids(s) for s in ids}
    check(ids and ids == want, f"tokenizer: {len(ids)} CosyVoice2 special tokens with the official ids (<|endofprompt|> = {ids.get('<|endofprompt|>')})")
    d0 = load_file(os.path.join(work, "dump_zero_shot.safetensors"))
    import torch as _t
    _t.manual_seed(0)
    noise = _t.randn([1, 80, 50 * 300])[0, :, :64].numpy()
    check(np.array_equal(noise, d0["flow_rand_noise_head"]), "flow noise: torch.manual_seed(0) + randn([1, 80, 15000]) is CausalConditionalCFM.rand_noise")

    # 2. stages
    for name in ("zero_shot", "cross_lingual", "instruct2"):
        for dt, tol in (("f32", "1e-4"), ("bf16", "3e-2")):
            rc, out = run([cli, "verify", "--model", tm[dt], "--dump", os.path.join(work, f"dump_{name}.safetensors"), "--device", "cpu", "--tol", tol])
            check(rc == 0 and "PASS" in out, f"verify {name} {dt} (tol {tol}): frontend, LLM, flow, HiFT")
            if rc != 0 or (dt == "f32" and name == "zero_shot"):
                print("\n".join("      " + l for l in out.splitlines() if "max|diff|" in l or "equal" in l or "agree" in l))

    # CAM++ segment pooling: the other mode too (onnxruntime ≤ 1.25 divides the last, incomplete segment by 100, newer ones and
    # PyTorch by the frames it holds); the reference is the converter's numpy CAM++, which reproduces onnxruntime in its mode
    pool = read_tmod_manifest(tm["f32"])["cosyvoice"].get("campplus_pool")
    other = "count" if pool == "window" else "window"
    tm_o = os.path.join(work, "cosy_pool.tmod")
    build(md, tm_o, dtype="f32", flow_dtype="f32", sha=False, campplus_pool=other)
    dz = dict(d0)
    dz["xvec"] = campplus_numpy(campplus_from_onnx(os.path.join(md, "campplus.onnx")), d0["fbank"].reshape(-1, 80), other)
    dump_o = os.path.join(work, "dump_pool.safetensors")
    save_file(dz, dump_o)
    rc, out = run([cli, "verify", "--model", tm_o, "--dump", dump_o, "--device", "cpu", "--tol", "1e-4"])
    line = next((l for l in out.splitlines() if "x-vector" in l), "")
    rel = float(np.max(np.abs(dz["xvec"] - d0["xvec"])) / np.max(np.abs(d0["xvec"])))
    check(line.rstrip().endswith("ok") and rel > 1e-3,
          f"CAM++ pooling: .tmod follows onnxruntime ('{pool}'); '{other}' also matches its numpy reference ({line.split('rel')[-1].split()[0] if 'rel' in line else line.strip()}; the modes differ by {rel:.1e})")

    # 3. whole utterances through the engine with the official tokens and noise
    cases = {}
    for name in ("zero_shot", "cross_lingual", "instruct2"):
        d = load_file(os.path.join(work, f"dump_{name}.safetensors"))
        inj = os.path.join(work, f"inj_{name}.safetensors")
        save_file({"0.tokens": d["gen_tokens"].astype(np.int32), "0.rand_ini": d["rand_ini"], "0.sine_noise": d["sine_noise"]}, inj)
        cases[name] = (d, inj)
    prompt = os.path.join(work, "prompt.wav")
    ptext = "Hello there, this is my voice."
    v_zs, v_cl = os.path.join(work, "zs.tvoice"), os.path.join(work, "cl.tvoice")
    rc, out = run([cli, "voice", "create", prompt, "--model", tm["f32"], "--text", ptext, "-o", v_zs])
    check(rc == 0, "voice from prompt.wav + its text (zero-shot)")
    rc, out = run([cli, "voice", "create", prompt, "--model", tm["f32"], "--xvector-only", "-o", v_cl])
    check(rc == 0, "voice from prompt.wav without text (cross-lingual)")
    feats = load_file(v_zs)
    tk = feats["feat.cosy_tokens"]
    check(np.array_equal(tk, d0["llm_prompt_tokens"]) and np.abs(feats["feat.cosy_xvec"] - d0["xvec"]).max() < 1e-4,
          f"voice features: {tk.size} speech tokens identical, x-vector within 1e-4 (torchaudio's resampler reproduced)")
    prof = os.path.join(work, "p.tvoice")
    run([cli, "profile", "new", prof, "--name", "tiny"])
    run([cli, "profile", "add", prof, "--voice", "anna", prompt, "--text", ptext])
    rc, out = run([cli, "profile", "set", prof, "--voice", "anna", "--style", "glad", "--instruct", "Speak happily"])
    check(rc == 0, "profile: style with an instruction")

    def say(text, voice, extra, *more):
        o = os.path.join(work, "o.wav")
        rc, so = run([cli, "say", text, "--model", tm["f32"], "--voice", voice, "-o", o, "--float", "--device", "cpu",
                      "--extra", json.dumps(extra), *more])
        return rc, so, (read_wav_f32(o) if rc == 0 else None)

    texts_of = json.load(open(os.path.join(work, "cases.json")))["texts"]
    for name, voice, more in (("zero_shot", v_zs, ()), ("cross_lingual", v_cl, ()), ("instruct2", prof, ("--style", "glad"))):
        d, inj = cases[name]
        rc, so, got = say(texts_of[name], voice, {"noise_file": inj}, *more)
        exp = d["audio"]
        ok = got is not None and len(got) == len(exp)
        err = float(np.abs(got - exp).max()) if ok else 9.9
        check(ok and err < 1e-3, f"{name}: engine == CosyVoice2Model.token2wav on the same tokens ({len(exp)} samples, max |diff| {err:.1e})"
              + ("" if rc == 0 else " — " + so[-300:]))

    # 4. text frontend vs text_normalize (no ttsfrd / wetext)
    import inflect
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    fe = types.SimpleNamespace(tokenizer=tok, allowed_special="all", text_frontend="", inflect_parser=inflect.engine())
    texts = [
        "I have 1234 apples and 7 pears. They cost 3.5 dollars!",
        "Short one.",
        " ".join(f"This is sentence number {i}, which talks about the weather and other things." for i in range(1, 12)),
        "He said \"Stop.\" Then he left; nobody followed: the end",
        "今天天气很好。我们去公园散步吧！你觉得怎么样？" * 4,
        "这是一个测试. 价格是3.5元 - 很便宜（真的）。",
        "Mixed text: 我们 are here 2024.",
        "...!!!",
    ]
    for t in texts:
        want = CosyVoiceFrontEnd.text_normalize(fe, t, split=True, text_frontend=True)
        rc, out = run([cli, "phonemize", "--model", tm["f32"], "--device", "cpu", t])
        got = json.loads(out)["pieces"] if rc == 0 else None
        check(got == want, f"text frontend: {t[:40]!r}… → {len(want)} piece(s)" + ("" if got == want else f"\n      vox:  {got}\n      want: {want}"))

    # 5. sampling, speed, streaming, markup, errors
    text = "Hello world. This is a test of the tiny model."
    a = say(text, v_zs, {}, "--seed", "7")
    b = say(text, v_zs, {}, "--seed", "7")
    c = say(text, v_zs, {}, "--seed", "8")
    check(a[0] == 0 and np.array_equal(a[2], b[2]), "own LLM sampling: same seed → identical audio")
    check(c[0] == 0 and (len(c[2]) != len(a[2]) or not np.array_equal(c[2], a[2])), "another seed → other audio")
    s = say(text, v_zs, {"speed": 1.5}, "--seed", "7")
    check(s[0] == 0 and len(s[2]) < len(a[2]), f"speed 1.5 → shorter ({len(s[2]) if s[2] is not None else 0} < {len(a[2])} samples)")
    rc, out = run([cli, "say", "Hello [laugh] there.", "--model", tm["f32"], "--voice", v_zs, "-o", os.path.join(work, "x.wav"), "--device", "cpu"])
    check(rc == 0, "[laugh] is spoken by the model as its own [laughter] token")
    rc, out = run([cli, "phonemize", "--model", tm["f32"], "--device", "cpu", "Hello [laugh] there."])
    check(rc == 0 and "[laughter]" in out, "[laugh] → [laughter] in the text the model reads")
    rc, out = run([cli, "voice", "speaker", "--model", tm["f32"], "--speaker", "x", "-o", os.path.join(work, "y.tvoice")])
    check(rc != 0, "no built-in speakers → clear error")

    try:
        import vox
        vox._native()
        has_py = True
    except Exception:  # noqa: BLE001
        has_py = False
    if has_py:
        with vox.Engine(tm["f32"], device="cpu") as eng:
            caps = eng.capabilities
            check(eng.info["family"] == "cosyvoice" and caps["clone_icl"] and caps["instruct"] and caps["speed"],
                  "Python: capabilities (cloning, instructions, speed)")
            v = eng.load_voice(v_zs)
            long = texts[2]
            w1 = eng.synthesize(long, v, seed=3)
            out = []
            with eng.session(v, seed=3) as s:
                for i in range(0, len(long), 37):
                    s.feed(long[i: i + 37])
                    out.extend(s.ready())
                s.end()
                out.extend(s)
                info = s.info
            check(np.array_equal(np.concatenate(out), w1), f"Python: text fed in pieces == whole text ({len(info['model']['pieces'])} pieces)")
            check(info["model"]["mode"] == "zero_shot", "session info: zero-shot mode, the pieces the model spoke")
            pv = vox.Voice.open(prof)
            with eng.session(pv, seed=3) as s:
                s.feed("[happy] Hi there.")
                s.end()
                list(s)
                info = s.info
            check(info["model"]["mode"] == "instruct2", "[happy] → instruct2 with the model's own wording of the emotion")
    else:
        print("SKIP: Python API checks (the vox package with the native module is not installed)")

    print("CosyVoice tiny:", "ALL OK" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
