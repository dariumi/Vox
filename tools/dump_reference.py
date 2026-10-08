#!/usr/bin/env python3
"""Эталонные дампы из оригинальной модели Qwen3-TTS для сверки бэкендов vox.

Для каждой фразы и seed сохраняет safetensors с промежуточными тензорами всех стадий:

    spk_mels            вход speaker encoder (мел-спектрограмма)       f32 [1, T, 128]
    spk_emb             эмбеддинг спикера                              f32 [D]
    talker_logits       логиты codec_head на каждом шаге (последняя поз.) f32 [steps, V]
    predictor_logits    логиты code predictor по кодбукам              f32 [steps, Q-1, V']
    codes               сгенерированные коды                           i32 [frames, Q]
    wav                 итоговое аудио 24 кГц                           f32 [N]

Как сверять: сэмплинг в vox (Philox) специально не совпадает с torch, поэтому по умолчанию дамп
жадный (do_sample=False). `vox verify --model M.tmod --dump case.safetensors` делает две проверки:
«с учителем» (в движок подаются эталонные коды, сравниваются логиты каждого шага и выход декодера)
и свободный жадный полёт через C ABI. Допуски: float32 ~1e-3 (отн.), bf16/q8 — около 5e-2.

    python tools/dump_reference.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --voice my_voice.tvoice --out ref_dumps/ --seeds 1 2 3

Без --voice используется --ref/--text (как в export_voice.py).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from vox.audio import load_wav  # noqa: E402
from vox.formats import VoiceProfile, save_safetensors  # noqa: E402
from weights import model_info  # noqa: E402

PHRASES = [
    "Привет! Это проверка движка синтеза речи.",
    "В 2026 году склад обработал 1 250 000 заказов, на 14% больше, чем год назад.",
    "Короткая фраза.",
    "Hello, this is a mixed language test, и немного русского текста.",
    "Вопрос? Восклицание! Перечисление: раз, два, три… и пауза.",
]


class Recorder:
    """Хуки на стадиях модели. Сохраняет только последнюю позицию логитов каждого вызова."""

    def __init__(self, model):
        self.reset()
        m = model
        self.handles = [m.talker.codec_head.register_forward_hook(self._talker)]
        if getattr(m, "speaker_encoder", None) is not None:  # CustomVoice/VoiceDesign checkpoints have none
            self.handles += [m.speaker_encoder.register_forward_pre_hook(self._spk_in),
                             m.speaker_encoder.register_forward_hook(self._spk_out)]
        for i, head in enumerate(m.talker.code_predictor.lm_head):
            self.handles.append(head.register_forward_hook(self._pred(i)))
        # Декодер кодека вызывается через обёртку speech_tokenizer.decode
        tok = m.speech_tokenizer
        orig = tok.decode

        def decode(*args, **kw):
            out = orig(*args, **kw)
            self.decoded.append((args, kw, out))
            return out
        tok.decode = decode

    def reset(self):
        self.spk_mels, self.spk_emb, self.talker, self.pred, self.decoded = [], [], [], [], []

    def _spk_in(self, mod, inp):
        self.spk_mels.append(inp[0].detach().float().cpu().numpy())

    def _spk_out(self, mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        self.spk_emb.append(o.detach().float().cpu().numpy())

    def _talker(self, mod, inp, out):
        self.talker.append(out[:, -1, :].detach().float().cpu().numpy()[0])

    def _pred(self, i):
        def hook(mod, inp, out):
            self.pred.append((i, out[:, -1, :].detach().float().cpu().numpy()[0]))
        return hook


def stack_predictor(pred, steps):
    """[(head_idx, logits)…] в порядке вызовов → [steps, Q-1, V]."""
    if not pred:
        return None
    nq = max(i for i, _ in pred) + 1
    rows, cur = [], [None] * nq
    for i, l in pred:
        cur[i] = l
        if i == nq - 1:
            rows.append(np.stack(cur))
            cur = [None] * nq
    return np.stack(rows[:steps]) if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", help=".tvoice из export_voice.py")
    ap.add_argument("--ref")
    ap.add_argument("--text")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--speaker", help="имя готового диктора (модель CustomVoice)")
    ap.add_argument("--instruct", help="инструкция о стиле (VoiceDesign / CustomVoice)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--phrases", help="файл с фразами, по одной на строку")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample", action="store_true", help="сэмплирование вместо жадного (для сверки свободного полёта нужен жадный)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

    model_dir, family, variant, sha = model_info(a.model)
    tts = Qwen3TTSModel.from_pretrained(model_dir, device_map=a.device, dtype=torch.float32)
    rec = Recorder(tts.model)

    mtype = getattr(tts.model, "tts_model_type", "base")
    named = mtype in ("custom_voice", "voice_design")
    if named:
        item = None
    elif a.voice:
        p = VoiceProfile.load(a.voice)
        if p.model.get("weights_sha256") not in ("", sha):
            raise SystemExit("voice was exported for other weights; export it again")
        item = VoiceClonePromptItem(
            ref_code=None if p.mode == "xvector_only" else torch.from_numpy(p.ref_codes.astype(np.int64)),
            ref_spk_embedding=torch.from_numpy(p.speaker_emb).to(tts.model.dtype),
            x_vector_only_mode=p.mode == "xvector_only", icl_mode=p.mode == "icl", ref_text=p.ref_text or None)
    else:
        if not a.ref:
            ap.error("--voice or --ref is required")
        item = tts.create_voice_clone_prompt(ref_audio=load_wav(a.ref), ref_text=a.text,
                                             x_vector_only_mode=not a.text)[0]

    phrases = [l.strip() for l in open(a.phrases, encoding="utf-8")] if a.phrases else PHRASES
    phrases = [p for p in phrases if p]
    os.makedirs(a.out, exist_ok=True)
    index = {"model": {"family": family, "variant": variant, "weights_sha256": sha}, "cases": []}

    for pi, text in enumerate(phrases):
        for seed in a.seeds:
            rec.reset()
            torch.manual_seed(seed)
            gen_kw = {} if a.sample else dict(do_sample=False, subtalker_dosample=False)
            if mtype == "custom_voice":
                wavs, sr = tts.generate_custom_voice(text=text, speaker=a.speaker, language=a.lang, instruct=a.instruct,
                                                     non_streaming_mode=False, max_new_tokens=a.max_frames, **gen_kw)
            elif mtype == "voice_design":
                wavs, sr = tts.generate_voice_design(text=text, instruct=a.instruct, language=a.lang,
                                                     non_streaming_mode=False, max_new_tokens=a.max_frames, **gen_kw)
            else:
                wavs, sr = tts.generate_voice_clone(text=text, language=a.lang, voice_clone_prompt=[item],
                                                    max_new_tokens=a.max_frames, **gen_kw)
            steps = len(rec.talker)
            tensors = {"wav": np.asarray(wavs[0], dtype=np.float32).reshape(-1)}
            if rec.talker:
                tensors["talker_logits"] = np.stack(rec.talker).astype(np.float32)
            pl = stack_predictor(rec.pred, steps)
            if pl is not None:
                tensors["predictor_logits"] = pl.astype(np.float32)
            if rec.spk_mels:
                tensors["spk_mels"] = rec.spk_mels[0].astype(np.float32)
                tensors["spk_emb"] = rec.spk_emb[0].reshape(-1).astype(np.float32)
            ref_frames = 0 if (item is None or item.ref_code is None) else int(item.ref_code.shape[0])
            if rec.decoded:
                args, kw, _ = rec.decoded[-1]
                enc = args[0] if args else kw.get("encoded")
                codes = enc[0]["audio_codes"] if isinstance(enc, (list, tuple)) else enc["audio_codes"]
                if hasattr(codes, "detach"):
                    codes = codes.detach().cpu().numpy()
                codes = np.asarray(codes).astype(np.int32).reshape(-1, codes.shape[-1])
                tensors["codes"] = codes[ref_frames:]  # только сгенерированные кадры (без кодов референса)
            name = f"case_{pi:02d}_seed{seed}.safetensors"
            save_safetensors(os.path.join(a.out, name), tensors,
                             {"text": text, "seed": str(seed), "sample_rate": str(sr), "language": a.lang,
                              "ref_frames": str(ref_frames), "generator": "qwen_tts/torch",
                              "voice": os.path.abspath(a.voice) if a.voice else "",
                              "speaker": a.speaker or "", "instruct": a.instruct or ""})
            index["cases"].append({"file": name, "text": text, "seed": seed, "steps": steps,
                                   "seconds": round(len(tensors["wav"]) / sr, 3)})
            print(f"{name}: {steps} steps, {len(tensors['wav']) / sr:.2f} s", file=sys.stderr)

    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
