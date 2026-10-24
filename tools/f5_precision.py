#!/usr/bin/env python3
"""How much each precision of the F5-TTS DiT changes the audio, on real weights.

    python tools/f5_precision.py --ckpt F5TTS_v1_Base/model_1250000.safetensors --vocab F5TTS_v1_Base/vocab.txt \\
        --vocos vocos-mel-24khz --ref ref.wav --ref-text "Text of the reference." --text "Привет, это проверка." \\
        --device vulkan:0 --work /mnt/data_ssd/f5prec [--nfe 32] [--seeds 42,7] [--variants f16,bf16,q8,...] [--asr]

Builds one .tmod per variant (f32 is the reference; existing files in --work are reused), synthesizes the same text with the
same seeds — so the starting noise is identical and only the weights differ — and compares every variant with f32:
  wave  relative L2 of the waveform (strict: a tiny shift of the ODE trajectory already moves phases),
  logS  mean |Δ| of the log-magnitude spectrogram in dB (what the ear compares),
  asr   what whisper hears (--asr; needs faster-whisper or openai-whisper).
Variants: f32, f16, bf16, q8 (bf16 + 8-bit block linears), and any of them with "+adaln", "+io", "+edges" (python -m
vox.build_f5 --keep-f32: those matrices stay f32), e.g. "q8+adaln", "bf16+adaln+io". Needs vox (pip install .) and numpy."""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_wav_f32(path):
    b = open(path, "rb").read()
    i, fmt = 12, None
    while i < len(b):
        cid, n = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", b[i + 8:i + 24])
        if cid == b"data":
            raw = b[i + 8:i + 8 + n]
            if fmt and fmt[0] == 3:
                return np.frombuffer(raw, "<f4").astype(np.float64)
            return np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
        i += 8 + n + (n & 1)
    raise ValueError(f"{path}: no data chunk")


def log_spec(x, n_fft=1024, hop=256):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    w = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * w
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)) + 1e-5)


def asr_fn():
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("small", device="auto", compute_type="auto")
        return lambda p: " ".join(s.text.strip() for s in m.transcribe(p)[0])
    except ImportError:
        pass
    try:
        import whisper
        m = whisper.load_model("small")
        return lambda p: m.transcribe(p)["text"].strip()
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--vocos", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--variants", default="f16,bf16,bf16+adaln,bf16+adaln+io,q8,q8+adaln,q8+adaln+io")
    ap.add_argument("--vox", default=os.path.join(ROOT, "build", "vox"), help="the vox CLI")
    ap.add_argument("--asr", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from vox.build_f5 import build

    os.makedirs(a.work, exist_ok=True)
    variants = ["f32"] + [v for v in a.variants.split(",") if v and v != "f32"]
    tmods = {}
    for v in variants:
        base, *keep = v.split("+")
        dtype, quant = ("bf16", "q8") if base == "q8" else (base, "")
        out = os.path.join(a.work, f"f5_{v.replace('+', '_')}.tmod")
        if not os.path.exists(out):
            print(f"building {v} …", file=sys.stderr)
            build(a.ckpt, a.vocab, a.vocos, out, dtype=dtype, quant=quant, keep_f32=",".join(keep), sha=False)
        tmods[v] = out
    voice = os.path.join(a.work, "ref.tvoice")
    subprocess.run([a.vox, "voice", "create", a.ref, "--model", tmods["f32"], "--text", a.ref_text, "-o", voice], check=True,
                   capture_output=True)
    asr = asr_fn() if a.asr else None
    if a.asr and asr is None:
        print("note: --asr needs faster-whisper or openai-whisper; skipped", file=sys.stderr)
    rows = []
    for seed in [int(s) for s in a.seeds.split(",") if s]:
        wav = {}
        for v in variants:
            o = os.path.join(a.work, f"out_{v.replace('+', '_')}_s{seed}.wav")
            r = subprocess.run([a.vox, "say", a.text, "--model", tmods[v], "--voice", voice, "-o", o, "--float", "--seed", str(seed),
                                "--device", a.device, "--extra", json.dumps({"nfe_step": a.nfe})], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{v}: synthesis failed: {r.stderr[-300:]}", file=sys.stderr)
                continue
            wav[v] = (o, read_wav_f32(o))
        ref = wav.get("f32")
        for v, (path, x) in wav.items():
            row = {"variant": v, "seed": seed, "MiB": os.path.getsize(tmods[v]) / 2**20}
            if ref is not None and v != "f32":
                y = ref[1]
                n = min(len(x), len(y))
                row["wave"] = float(np.linalg.norm(x[:n] - y[:n]) / (np.linalg.norm(y[:n]) + 1e-12)) if len(x) == len(y) else float("nan")
                row["logS"] = float(np.mean(np.abs(log_spec(x[:n]) - log_spec(y[:n]))))
            if asr:
                row["asr"] = asr(path)
            rows.append(row)
    print(f"\n| variant | seed | .tmod MiB | wave rel L2 vs f32 | log-spectrum mean abs dB vs f32 |" + (" ASR |" if asr else ""))
    print("|---|---|---|---|---|" + ("---|" if asr else ""))
    for r in rows:
        w = f"{r['wave']:.2e}" if "wave" in r else "—"
        s = f"{r['logS']:.2f}" if "logS" in r else "—"
        print(f"| {r['variant']} | {r['seed']} | {r['MiB']:.0f} | {w} | {s} |" + (f" {r.get('asr', '')} |" if asr else ""))


if __name__ == "__main__":
    main()
