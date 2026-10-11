#!/usr/bin/env python3
"""Сверка vox с официальным F5-TTS на НАСТОЯЩИХ весах (в песочнице их нет, поэтому запускать у себя).

    python tools/check_f5_real.py --ckpt F5TTS_v1_Base/model_1250000.safetensors --vocab F5TTS_v1_Base/vocab.txt \\
        --vocos vocos-mel-24khz --audio ref.wav --ref-text "Текст, который сказан в ref.wav" \\
        --text "Фраза для озвучки." [--dtypes f32,bf16,q8] [--device auto] [--nfe 32]

Что делает: официальный `infer_batch_process` (f5_tts + vocos) и `vox say` получают один и тот же образец, текст и шум
(шум из официального прогона подставляется в vox через extra_json), затем сравниваются mel и звук. Для f32 ожидается
совпадение порядка 1e-5…1e-6; для bf16/q8 — отклонение квантования (смотрите относительную ошибку и на слух).
Образец сначала проходит официальную подготовку (`preprocess_ref_audio_text`: обрезка тишины, ≤ 12 с), vox получает уже
подготовленный файл. Нужны: torch, torchaudio, f5-tts, vocos, safetensors, pydub (как у самого F5-TTS), собранный build/vox.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_wav(path):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--vocos", required=True, help="папка charactr/vocos-mel-24khz (config.yaml + pytorch_model.bin)")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--dtypes", default="f32,bf16,q8")
    ap.add_argument("--device", default="auto", help="устройство vox: auto | cpu | vulkan:0 | …")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--model-cfg", default="F5TTS_v1_Base", help="имя конфигурации из f5_tts/configs (F5TTS_v1_Base, F5TTS_Base…)")
    ap.add_argument("--cli", default=os.path.join(ROOT, "build", "vox"))
    ap.add_argument("--threads", type=int, default=0, help="потоки для vox (0 = по умолчанию) и torch")
    ap.add_argument("--vox-model-cfg", default="", choices=["", "v0", "v1"],
                    help="конфиг F5 для конвертера vox; по умолчанию берётся из arch официальной конфигурации (--model-cfg)")
    ap.add_argument("--keep", default="", help="папка, куда сложить tmod, wav и шум (по умолчанию временная)")
    ap.add_argument("--tokenizer", default="pinyin")
    ap.add_argument("--arch", default="", help="JSON с параметрами DiT вместо конфигурации (для крошечной тестовой модели)")
    ap.add_argument("--dim-head", type=int, default=64)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # вывод не теряется, если прогон оборвёт таймаут
    except Exception:  # noqa: BLE001
        pass

    try:
        import datasets  # noqa: F401
    except Exception:  # обучающий код f5_tts тянет datasets; для инференса он не нужен — подменяем модуль набора данных
        import types
        stub = types.ModuleType("f5_tts.model.dataset")
        stub.DynamicBatchSampler = type("DynamicBatchSampler", (), {})
        stub.collate_fn = lambda *x, **k: None
        sys.modules["f5_tts.model.dataset"] = stub
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from importlib.resources import files
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer import utils_infer as U
    from f5_tts.model import CFM
    from f5_tts.model.utils import get_tokenizer

    work = a.keep or tempfile.mkdtemp(prefix="voxf5real_")
    os.makedirs(work, exist_ok=True)
    cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{a.model_cfg}.yaml")))
    backbone = get_class(f"f5_tts.model.{cfg.model.backbone}")
    arch = OmegaConf.create(json.loads(a.arch)) if a.arch else cfg.model.arch
    mel = cfg.model.mel_spec
    # то, чего нет в весах: из arch берём text_mask_padding и pe_attn_head (F5TTS_Base: False и 1; F5TTS_v1_Base: True и null)
    tmp_flag = bool(arch.get("text_mask_padding", True))
    pe_head = arch.get("pe_attn_head", None)
    vox_dev = a.device + (f",threads={a.threads}" if a.threads else "")
    vox_cfg = a.vox_model_cfg or ("v0" if pe_head else "v1")
    print(f"vox F5 config: {vox_cfg} (text_mask_padding={tmp_flag}, pe_attn_head={pe_head})")
    vocab_map, vocab_size = get_tokenizer(a.vocab, "custom")
    model = CFM(transformer=backbone(**arch, text_num_embeds=vocab_size, mel_dim=mel.n_mel_channels),
                mel_spec_kwargs=dict(n_fft=mel.n_fft, hop_length=mel.hop_length, win_length=mel.win_length,
                                     n_mel_channels=mel.n_mel_channels, target_sample_rate=mel.target_sample_rate,
                                     mel_spec_type=mel.mel_spec_type),
                odeint_kwargs=dict(method="euler"), vocab_char_map=vocab_map)
    from f5_tts.infer.utils_infer import load_checkpoint
    model = load_checkpoint(model, a.ckpt, "cpu", use_ema=True).eval()
    vocoder = U.load_vocoder("vocos", is_local=True, local_path=a.vocos, device="cpu")

    ref_audio, ref_text = U.preprocess_ref_audio_text(a.audio, a.ref_text)
    import torchaudio
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * a.speed)
    batches = U.chunk_text(a.text, max_chars=max_chars)
    print(f"official: ref {audio.shape[-1] / sr:.2f} s, {len(batches)} chunk(s), nfe {a.nfe}")

    class Serial:
        def __init__(self, *x, **k): pass
        def __enter__(self): return self
        def __exit__(self, *x): return False
        def submit(self, fn, *x, **k):
            class F:
                def __init__(s, v): s.v = v
                def result(s): return s.v
            return F(fn(*x, **k))
    U.ThreadPoolExecutor = Serial
    noises = []
    orig = torch.randn

    def fake(*s, **k):
        if len(s) == 2 and s[1] == mel.n_mel_channels:
            r = orig(*s, **k)
            noises.append(r.numpy().astype(np.float32).copy())
            return r
        return orig(*s, **k)
    torch.manual_seed(1234)
    torch.randn = fake
    t0 = time.time()
    try:
        wave, _, spec = next(U.infer_batch_process((audio, sr), ref_text, batches, model, vocoder, progress=None, nfe_step=a.nfe,
                                                   cfg_strength=a.cfg, sway_sampling_coef=a.sway, speed=a.speed, device="cpu"))
    finally:
        torch.randn = orig
    t_off = time.time() - t0
    from safetensors.numpy import save_file
    noise_path = os.path.join(work, "noise.safetensors")
    save_file({f"noise.{i}": np.ascontiguousarray(n) for i, n in enumerate(noises)}, noise_path)
    print(f"official: {len(wave) / sr:.2f} s of audio in {t_off:.1f} s (CPU, torch)")

    sys.path.insert(0, os.path.join(ROOT, "python"))
    from vox.build_f5 import build
    ok = True
    for dt in a.dtypes.split(","):
        dtype, quant = ("f32", "q8") if dt == "q8" else (dt, "")
        tmod = os.path.join(work, f"f5_{dt}.tmod")
        build(a.ckpt, a.vocab, a.vocos, tmod, dtype=dtype, quant=quant, tokenizer=a.tokenizer, variant=a.model_cfg, dim_head=a.dim_head, sha=False,
              model_cfg=vox_cfg, text_mask_padding=tmp_flag, pe_attn_head=(int(pe_head) if pe_head else None))
        voice = os.path.join(work, f"v_{dt}.tvoice")
        r = subprocess.run([a.cli, "voice", "create", ref_audio, "--model", tmod, "--text", a.ref_text, "-o", voice, "--device", vox_dev],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr)
            return 1
        out = os.path.join(work, f"out_{dt}.wav")
        dump = os.path.join(work, f"mel_{dt}.safetensors")
        t0 = time.time()
        r = subprocess.run([a.cli, "say", a.text, "--model", tmod, "--voice", voice, "-o", out, "--float", "--device", vox_dev, "--nfe", str(a.nfe),
                            "--cfg", str(a.cfg), "--sway", str(a.sway), "--speed", str(a.speed),
                            "--extra", json.dumps({"noise_file": noise_path, "dump_file": dump, "max_chars": max_chars})],
                           capture_output=True, text=True)
        t_vox = time.time() - t0
        if r.returncode:
            print(r.stdout + r.stderr)
            return 1
        got = read_wav(out)
        # токены: если словарь не совпал с текстом (как было с CRLF в vocab.txt), все символы становятся 0 (= пробел)
        from safetensors.numpy import load_file as _lf
        dd = _lf(dump)
        idx = sorted(int(k.split(".")[1]) for k in dd if k.startswith("mel."))
        ids = np.concatenate([dd[f"ids.{i}"] for i in idx]) if idx else np.zeros(0)
        n_sp = sum(b.count(" ") for b in batches) + len(a.ref_text) - len(a.ref_text.replace(" ", ""))
        if ids.size and (np.all(ids == 0) or np.count_nonzero(ids == 0) > 2 * n_sp + 4):
            ok = False
            print(f"  ! {np.count_nonzero(ids == 0)} из {ids.size} токенов = 0: символы текста не найдены в словаре (кодировка или переводы строк vocab.txt?)")
        # mel: та же величина, что сравнивает официальный infer_batch_process до вокодера
        vm = np.concatenate([dd[f"mel.{i}"] for i in idx], 0).T if idx else np.zeros((0, 0))
        om = np.asarray(spec, dtype=np.float32)
        if vm.shape == om.shape:
            mel_rel = float(np.linalg.norm(vm - om) / (np.linalg.norm(om) + 1e-12))
            print(f"  mel: {vm.shape[1]} кадров, относительная L2 {mel_rel:.3e}")
        else:
            mel_rel = 9.9
            print(f"  mel: другая форма {vm.shape} против официальной {om.shape}")
        # звук со сдвигом: сдвиг выдаёт неверную обрезку краёв ISTFT (padding вокодера)
        w = np.asarray(wave, dtype=np.float64)
        g = np.asarray(got, dtype=np.float64)
        L = 1 << int(np.ceil(np.log2(len(w) + len(g))))
        xc = np.fft.irfft(np.fft.rfft(g, L) * np.conj(np.fft.rfft(w, L)), L)
        lags = np.r_[0:2048, L - 2048:L]
        best = int(lags[np.argmax(xc[lags])])
        shift = best if best < L // 2 else best - L
        gs = g[shift:] if shift >= 0 else np.r_[np.zeros(-shift), g]
        na = min(len(gs), len(w))
        rel_al = float(np.linalg.norm(gs[:na] - w[:na]) / (np.linalg.norm(w[:na]) + 1e-12))
        print(f"  звук: сдвиг {shift} отсч., после выравнивания относительная L2 {rel_al:.3e}" + ("" if shift == 0 else "  ← сдвиг: проверьте padding вокодера (config.yaml)"))
        n = min(len(got), len(wave))
        rel = float(np.linalg.norm(got[:n] - wave[:n]) / (np.linalg.norm(wave[:n]) + 1e-12))
        mx = float(np.abs(got[:n] - wave[:n]).max())
        d = np.asarray(wave)
        print(f"vox {dt:5s}: {len(got) / sr:.2f} s (official {len(wave) / sr:.2f} s), relative L2 error {rel:.3e}, max|diff| {mx:.3e} "
              f"(max|ref| {np.abs(d).max():.3f}), {t_vox:.1f} s wall [{a.device}]")
        # главный критерий — mel (выход DiT): Vocos восстанавливает фазу, и малые расхождения mel дают заметно большие в форме волны
        # (f32: mel 4e-6 → звук ~2e-4 на hotstone228). Звук: для f32 — жёсткий порог, для bf16/q8 только печатается.
        mel_lim, lim = (1e-4, 1e-3) if dt == "f32" else (0.05, float("inf"))
        if len(got) != len(wave) or rel > lim or mel_rel > mel_lim:
            ok = False
            print(f"  ^ вне допуска (mel {mel_lim:g}, звук {lim:g}) или другая длина")
    print("PASS" if ok else "FAIL", "— файлы:", work)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
