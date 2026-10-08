#!/usr/bin/env python3
"""Q4-веса: сверка ядер CPU и Vulkan с эталоном «официальная модель с теми же квантованными весами».

Идея: берём крошечную официальную модель, прогоняем её линейные слои через quantize_q4 → распаковку (ровно те, что
конвертер кладёт в Q4) и сохраняем как обычный fp32-чекпойнт. Официальный PyTorch на таком чекпойнте даёт дамп логитов.
Движок с настоящим Q4 (.tmod --quant q4) должен сойтись с ним: на GPU (веса распаковываются, активации float) — как fp32,
на CPU — с поправкой на int8-квантование активаций (распределения, KL).
Показывает корректность упаковки/ядер, но НЕ качество Q4 на настоящей модели — его замеряют на своих голосах."""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
GGUF = os.environ.get("VOX_QWEN2_GGUF", "")
DEVICES = [d for d in os.environ.get("VOX_TEST_DEVICES", "cpu").split(";") if d]
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))


def run(cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=ENV)
    if check and r.returncode:
        print("$", " ".join(map(str, cmd)), "\n", r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(1)
    return r


def main(cli):
    if not GGUF or not os.path.exists(GGUF):
        print("SKIP: set VOX_QWEN2_GGUF")
        return 0
    try:
        import torch  # noqa: F401
        import qwen_tts  # noqa: F401
    except ImportError as e:
        print("SKIP:", e)
        return 0
    sys.path.insert(0, os.path.join(ROOT, "python"))
    return sum(one(cli, q) for q in os.environ.get("VOX_TEST_QUANTS", "q4,q4mlp,q4full").split(","))


def one(cli, quant):
    print(f"=== --quant {quant} ===")
    from vox.build import LINEAR_Q8, quant_kind, quantize_q4, quantize_q8
    from vox.formats import load_safetensors, save_safetensors
    from vox.audio import save_wav

    work = tempfile.mkdtemp(prefix=f"vox_{quant}_")
    model, sim = os.path.join(work, "model"), os.path.join(work, "sim")
    run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_official.py"), "--vocab", GGUF, "--out", model])
    shutil.copytree(model, sim)
    tensors, meta = load_safetensors(os.path.join(model, "model.safetensors"))
    n = n4 = 0
    for name, w in list(tensors.items()):
        kind = quant_kind(name, quant) if (w.ndim == 2 and w.shape[1] % 32 == 0) else ""
        if kind == "q4":
            packed, d = quantize_q4(w)
            lo, hi = packed & 15, packed >> 4
            codes = np.concatenate([lo.reshape(w.shape[0], -1, 16), hi.reshape(w.shape[0], -1, 16)], axis=2)
            tensors[name] = ((codes.astype(np.float32) - 8.0) * d[..., None]).reshape(w.shape).astype(np.float32)
            n += 1
            n4 += 1
        elif kind == "q8":
            q, sc = quantize_q8(w)
            tensors[name] = (q.reshape(w.shape[0], -1, 32).astype(np.float32) * sc[..., None]).reshape(w.shape).astype(np.float32)
            n += 1
    assert n >= 20 and n4 >= 4, (n, n4)
    save_safetensors(os.path.join(sim, "model.safetensors"), tensors, meta)
    print(f"{n} linear layers replaced by their quantised round trip ({n4} of them Q4)")

    q4 = os.path.join(work, "tiny_%s.tmod" % quant)
    run([sys.executable, "-m", "vox.build", "--model", model, "-o", q4, "--dtype", "f32", "--quant", quant])
    f32 = os.path.join(work, "tiny_f32.tmod")
    run([sys.executable, "-m", "vox.build", "--model", model, "-o", f32, "--dtype", "f32"])
    print("size: f32 %.1f KiB, q4 %.1f KiB" % (os.path.getsize(f32) / 1024, os.path.getsize(q4) / 1024))
    assert os.path.getsize(q4) < os.path.getsize(f32)

    sr = 24000
    t = np.arange(int(sr * 3.3)) / sr
    save_wav(os.path.join(work, "ref.wav"), (0.3 * np.sin(2 * np.pi * np.cumsum(150 + 60 * np.sin(2 * np.pi * 0.3 * t)) / sr)).astype(np.float32), sr)
    voice = os.path.join(work, "v.tvoice")
    run([sys.executable, os.path.join(ROOT, "tools", "export_voice.py"), "--model", sim, "--ref", os.path.join(work, "ref.wav"),
         "--text", "Привет, это референс голоса для проверки.", "--lang", "english", "-o", voice])
    phrase = os.path.join(work, "p.txt")
    open(phrase, "w", encoding="utf-8").write("Hello, this is a test.\n")
    dump = os.path.join(work, "dump")
    run([sys.executable, os.path.join(ROOT, "tools", "dump_reference.py"), "--model", sim, "--voice", voice, "--lang", "english",
         "--seeds", "1", "--max-frames", "30", "--phrases", phrase, "--out", dump])
    d = os.path.join(dump, "case_00_seed1.safetensors")

    failures = 0
    for dev in DEVICES:
        gpu = dev.startswith("vulkan") and "layers=" not in dev  # гибрид: часть слоёв на CPU с int8-активациями
        # GPU: активации остаются float → почти как fp32; CPU: int8-активации → сравнение распределений
        cmd = [cli, "verify", "--device", dev, "--model", q4, "--dump", d, "--voice", voice]
        cmd += ["--tol", "2e-3"] if gpu else ["--quant", "1", "--kl-tol", "0.05", "--top5-min", "0.9"]
        if quant == "q4full" and gpu:
            cmd += ["--free", "0"]  # логиты совпадают до 1e-6; свободный полёт на случайных весах крошечной модели может перевернуть почти равный argmax
        r = run(cmd, check=False)
        ok = r.returncode == 0
        print(("PASS " if ok else "FAIL ") + f"verify {quant} on {dev}")
        if not ok:
            print(r.stdout[-2500:], r.stderr[-1500:])
            failures += 1
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")) else 0)
