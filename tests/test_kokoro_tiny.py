"""Kokoro in vox against the official `kokoro` package on a tiny model with random weights (tools/make_tiny_kokoro.py).
Run: python tests/test_kokoro_tiny.py [build/vox]   (needs torch, kokoro, misaki; otherwise SKIP)

What is checked:
  1. every stage (`vox verify`): token ids, voice pack row, PL-BERT, duration encoder, durations, F0 / N, text encoder,
     alignment, the harmonic source and its spectrogram, the iSTFTNet generator — f32 tight, bf16 within quantization;
  2. whole utterances vs KPipeline with the same source noise: the text → phoneme pieces of the Python frontend (misaki)
     must equal KPipeline's, the audio must have the same length and the same spectral envelope (log-mel). Sample-exact
     equality is impossible by design: the NSF source integrates F0 over the utterance, so the ~1e-6 float differences of F0
     shift the waveform's phase (the decoder alone, fed the reference F0, matches to ~1e-6 — part 1);
  3. the built-in English G2P (misaki's lexicons, no spaCy / espeak) against misaki on tests/data/g2p_en.txt;
  4. seeds, speed, voice mixing, long phoneme input (split at 510), errors, Python API.
Not checked: quality with the real weights (no Hugging Face access during development); GPU (Kokoro runs on the CPU)."""
import json
import os
import struct
import subprocess
import sys
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)

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


def mel_distance(a, b):
    """relative L2 of mel power spectrograms and mean |Δ| in dB of log-mel (1024 / 256, 80 mels)"""
    import torch
    import torchaudio
    m = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80)
    A, B = (m(torch.from_numpy(np.ascontiguousarray(x, np.float32))).numpy() for x in (a, b))
    n = min(A.shape[1], B.shape[1])
    A, B = A[:, :n], B[:, :n]
    return float(np.linalg.norm(A - B) / np.linalg.norm(B)), float(np.mean(np.abs(np.log10(A + 1e-5) - np.log10(B + 1e-5))) * 10)


def main(cli):
    try:
        import torch  # noqa: F401
        import kokoro  # noqa: F401
        import misaki.en  # noqa: F401
        import torchaudio  # noqa: F401
        from safetensors.numpy import load_file
    except Exception as e:  # noqa: BLE001
        print("SKIP: needs torch, torchaudio, kokoro, misaki:", e)
        return 0
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from vox.build_kokoro import build
    from vox import kokoro_frontend as kf
    sys.path.remove(os.path.join(ROOT, "python"))
    for m in [m for m in sys.modules if m == "vox" or m.startswith("vox.")]:
        del sys.modules[m]

    work = tempfile.mkdtemp(prefix="voxkokoro_")
    rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_kokoro.py"), "--out", work])
    assert rc == 0, out[-3000:]
    tm = {}
    for dt in ("f32", "bf16"):
        tm[dt] = os.path.join(work, f"kokoro_{dt}.tmod")
        build(os.path.join(work, "config.json"), os.path.join(work, "kokoro-tiny.pth"), tm[dt], os.path.join(work, "voices"), dt, sha=False)
    os.remove(os.path.join(work, "kokoro-tiny.pth"))   # 290 MB: not needed any more
    e2e = json.load(open(os.path.join(work, "e2e.json"), encoding="utf-8"))
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures += 1

    # 1. stages
    for dt, tol in (("f32", "1e-4"), ("bf16", "2e-2")):
        rc, out = run([cli, "verify", "--model", tm[dt], "--dump", os.path.join(work, "dump.safetensors"), "--device", "cpu", "--tol", tol])
        check(rc == 0 and "PASS" in out, f"verify {dt} (tol {tol}): every stage of KModel.forward")
        if rc != 0 or dt == "f32":
            print("\n".join("      " + l for l in out.splitlines() if "max|diff|" in l))

    # 2. whole utterances
    voices = {}
    for v in ("af_tiny", "bm_tiny"):
        voices[v] = os.path.join(work, v + ".tvoice")
        rc, out = run([cli, "voice", "speaker", "--model", tm["f32"], "--speaker", v, "-o", voices[v]])
        assert rc == 0, out
    for name, sc in e2e["scenarios"].items():
        ref_chunks = [c["phonemes"] for c in sc["chunks"]]
        if not sc["phonemes_in"]:
            mine = kf.frontend(False).chunks(sc["text"])
            check(mine == ref_chunks, f"{name}: Python frontend (misaki) gives KPipeline's {len(ref_chunks)} piece(s) of phonemes")
        ref = load_file(os.path.join(work, f"e2e_{name}.safetensors"))
        exp = np.concatenate([ref[f"{i}.audio"] for i in range(len(ref_chunks))])
        outp = os.path.join(work, f"out_{name}.wav")
        rc, so = run([cli, "say", "\n".join(ref_chunks), "--model", tm["f32"], "--voice", voices[sc["voice"]], "-o", outp, "--float",
                      "--device", "cpu", "--extra", json.dumps({"phonemes": True, "speed": sc["speed"],
                                                                "noise_file": os.path.join(work, f"e2e_{name}.safetensors")})])
        assert rc == 0, so
        got = read_wav_f32(outp)
        rel, db = mel_distance(got, exp) if len(got) == len(exp) else (9.9, 99.0)
        check(len(got) == len(exp) and rel < 0.08 and db < 0.8,
              f"{name}: {len(ref_chunks)} piece(s), {len(exp)} samples (same length), mel rel {rel:.3f}, log-mel {db:.2f} dB vs KPipeline")
    # the same text, other noise: how far "different" is (the thresholds above sit well below it)
    sc = e2e["scenarios"]["phonemes"]
    outp = os.path.join(work, "out_other_noise.wav")
    run([cli, "say", sc["text"], "--model", tm["f32"], "--voice", voices["af_tiny"], "-o", outp, "--float", "--device", "cpu",
         "--extra", json.dumps({"phonemes": True})])
    exp = load_file(os.path.join(work, "e2e_phonemes.safetensors"))["0.audio"]
    rel, db = mel_distance(read_wav_f32(outp), exp)
    check(rel > 0.1, f"for scale: own noise instead of the reference's gives mel rel {rel:.3f}, {db:.2f} dB")

    # 3. built-in G2P vs misaki
    lines = [l.strip() for l in open(os.path.join(HERE, "data", "g2p_en.txt"), encoding="utf-8") if l.strip()]
    from misaki import en
    for brit in (False, True):
        g = en.G2P(trf=False, british=brit, fallback=(lambda t: ("", 1)), unk="")
        p = subprocess.run([cli, "phonemize", "--model", tm["f32"], "--device", "cpu", "--json", "--lang", "en-gb" if brit else "en-us"],
                           input="\n".join(json.dumps(l) for l in lines) + "\n", capture_output=True, text=True)
        outs = [json.loads(x)["phonemes"] for x in p.stdout.splitlines()]
        same = sum(1 for l, o in zip(lines, outs) if g(l)[0].strip() == o.strip())
        words = sum(sum(1 for a, b in zip(g(l)[0].split(), o.split()) if a == b) for l, o in zip(lines, outs))
        total = sum(max(len(g(l)[0].split()), len(o.split())) for l, o in zip(lines, outs))
        check(len(outs) == len(lines) and words / total >= 0.9,
              f"built-in G2P ({'en-gb' if brit else 'en-us'}) vs misaki: {same}/{len(lines)} sentences identical, {100 * words / total:.1f}% words "
              f"(misaki here has no espeak: unknown words are empty there, spelled out by vox)")
    for case, want in e2e["g2p"].items():
        rc, out = run([cli, "phonemize", "--model", tm["f32"], "--device", "cpu", case])
        check(out.strip() == want["us"].strip(), f"G2P: {case!r}")

    # 4. seeds, speed, mixing, long input, errors
    text = "Hello world. This is a test of the tiny model."
    def say(extra, voice="af_tiny", txt=text):
        o = os.path.join(work, "s.wav")
        rc, so = run([cli, "say", txt, "--model", tm["f32"], "--voice", voices.get(voice, voice), "-o", o, "--float", "--device", "cpu",
                      "--extra", json.dumps(extra)])
        return rc, so, (read_wav_f32(o) if rc == 0 else None)
    a, b, c = say({}), say({}), say({"seed": 0})
    rc, so, d = say({"speed": 1.4})
    check(a[0] == 0 and np.array_equal(a[2], b[2]), "same seed → identical audio")
    check(d is not None and len(d) < len(a[2]), f"speed 1.4 → shorter ({len(d) if d is not None else 0} < {len(a[2])} samples)")
    mix = os.path.join(work, "mix.tvoice")
    rc, out = run([cli, "voice", "speaker", "--model", tm["f32"], "--speaker", "af_tiny,bm_tiny", "-o", mix])
    rc2, so, m = say({}, voice=mix)
    check(rc == 0 and rc2 == 0 and m is not None and not np.array_equal(m, a[2]), "voice mix af_tiny,bm_tiny (packs averaged)")
    rc, out = run([cli, "voice", "speaker", "--model", tm["f32"], "--speaker", "zz_nope", "-o", os.path.join(work, "x.tvoice")])
    check(rc != 0 and "unknown speaker" in out, "unknown voice → clear error")
    long_ps = " ".join(["ðɪs ɪz ɐ lˈɔŋ sˈɛntəns,"] * 40)
    rc, so, l = say({"phonemes": True}, txt=long_ps)
    check(rc == 0 and l is not None and len(l) > 0, f"phonemes longer than 510 are split into pieces ({len(long_ps)} phonemes)")
    rc, out = run([cli, "say", "Hola", "--model", tm["f32"], "--voice", voices["af_tiny"], "-o", os.path.join(work, "x.wav"),
                   "--extra", json.dumps({"lang": "es"})])
    check(rc != 0 and "phonemes" in out, "lang other than English → clear error (pass phonemes)")

    # Python API (installed package with the native module)
    try:
        import vox
        vox._native()
        has_py = True
    except Exception:  # noqa: BLE001
        has_py = False
    if has_py:
        with vox.Engine(tm["f32"], device="cpu") as eng:
            caps = eng.capabilities
            check(eng.info["family"] == "kokoro" and caps["named_speakers"] and "af_tiny" in eng.speakers, f"Python: capabilities {eng.speakers}")
            r = eng.phonemize("Hello world! This is a tiny test.")
            check(r["chunks"] == [c["phonemes"] for c in e2e["scenarios"]["short"]["chunks"]] and r["g2p"] == "misaki",
                  "Python: Engine.phonemize uses misaki → KPipeline's pieces")
            rb = eng.phonemize("Hello world! This is a tiny test.", g2p="builtin")
            check(rb["g2p"] == "builtin" and len(rb["chunks"]) == 1, f"Python: built-in G2P → {rb['chunks'][0]!r}")
            v = eng.speaker_voice("af_tiny")
            w1 = eng.synthesize("Hello world! This is a tiny test.", v, seed=3)
            ps = "\n".join(r["chunks"])
            w2 = eng.synthesize(ps, v, seed=3, phonemes=True)
            check(np.array_equal(w1, w2), "Python: text through misaki == the same phonemes given directly")
            out = []
            with eng.session(v, seed=3) as s:
                for piece in ["Hello wor", "ld! This is ", "a tiny test."]:
                    s.feed(piece)
                s.end()
                out.extend(s)
            check(np.array_equal(np.concatenate(out), w1), "Python: text fed in pieces == whole text")
    else:
        print("SKIP: Python API checks (the vox package with the native module is not installed)")
    print("Kokoro tiny: " + ("ALL OK" if failures == 0 else f"{failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
