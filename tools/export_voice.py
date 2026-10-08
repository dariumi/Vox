#!/usr/bin/env python3
"""Создаёт .tvoice из оригинального пайплайна Qwen3-TTS (пакет qwen-tts).

Нужен, пока у vox нет собственного бэкенда qwen3-tts: профили, сделанные этим
скриптом, будут работать в движке без пересоздания, потому что содержат ровно те
тензоры, что оригинальная модель кладёт в voice clone prompt.

    pip install qwen-tts torch
    python tools/export_voice.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --ref ref.wav --text "Текст, который звучит в референсе" \\
        --name my_voice --lang ru --seed 42 -o my_voice.tvoice

Режим --xvector-only сохраняет только эмбеддинг спикера (без текста и кодов референса).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from vox.audio import load_wav, save_wav  # noqa: E402
from vox.formats import VoiceProfile  # noqa: E402
from weights import model_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id или локальная папка модели Base")
    ap.add_argument("--ref", required=True, help="референс WAV (5–15 с, чистая речь)")
    ap.add_argument("--text", default=None, help="текст референса (обязателен в режиме ICL)")
    ap.add_argument("--xvector-only", action="store_true")
    ap.add_argument("--name", default="voice")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    if not a.xvector_only and not a.text:
        ap.error("--text is required in ICL mode (or pass --xvector-only)")

    import torch
    from qwen_tts import Qwen3TTSModel

    model_dir, family, variant, sha = model_info(a.model)
    print(f"model: {variant}  weights sha256: {sha[:16]}…", file=sys.stderr)
    tts = Qwen3TTSModel.from_pretrained(model_dir, device_map=a.device, dtype=torch.float32)

    pcm, sr = load_wav(a.ref)
    items = tts.create_voice_clone_prompt(ref_audio=(pcm, sr), ref_text=a.text, x_vector_only_mode=a.xvector_only)
    it = items[0]

    emb = it.ref_spk_embedding.detach().float().cpu().numpy().reshape(-1)
    prof = VoiceProfile(speaker_emb=emb, name=a.name, language=a.lang,
                        mode="xvector_only" if a.xvector_only else "icl",
                        model={"family": family, "variant": variant, "weights_sha256": sha})
    if not a.xvector_only:
        codes = it.ref_code.detach().cpu().numpy()
        if codes.ndim == 1:
            raise SystemExit("this exporter supports the 12Hz (multi-codebook) models only")
        if codes.max() > np.iinfo(np.int16).max:
            raise SystemExit("codebook ids do not fit into int16")
        prof.ref_codes = codes.astype(np.int16)
        prof.ref_text = a.text
        # Токены ровно в том виде, в каком референс подаётся модели (с шаблоном чата).
        ids = tts._tokenize_texts([tts._build_ref_text(a.text)])[0]
        prof.ref_text_tokens = ids.detach().cpu().numpy().reshape(-1).astype(np.int32)

    tmp = a.out + ".ref.wav"
    save_wav(tmp, pcm, sr)
    prof.ref_audio = open(tmp, "rb").read()
    os.remove(tmp)

    if a.seed is not None:
        prof.sampling["seed"] = a.seed
    if a.temperature is not None:
        prof.sampling["temperature"] = a.temperature
    prof.save(a.out)
    print(f"saved {a.out}: emb {emb.shape[0]}, "
          f"{'no codes' if a.xvector_only else f'codes {prof.ref_codes.shape}'}", file=sys.stderr)


if __name__ == "__main__":
    main()
