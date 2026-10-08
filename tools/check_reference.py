#!/usr/bin/env python3
"""Проверка референса для ICL-клонирования ДО экспорта голоса.

Лишние слова в начале синтеза («хвост референса») чаще всего значат, что аудио и ref_text не совпадают: аудио обрезано
посреди слова, в тексте есть лишнее/нет сказанного, в конце нет паузы. Скрипт:
  * считает длительность, темп речи (знаков/с) и тишину в начале/конце;
  * если установлен faster-whisper или openai-whisper — расшифровывает аудио и сравнивает с --ref-text (WER),
    показывает, где текст расходится с речью;
  * --write-trimmed out.wav — обрезает по границе последнего целиком сказанного слова/предложения (нужен whisper с
    таймстемпами слов) и добавляет 0.25 с тишины в конец (ICL любит, когда референс заканчивается паузой).

    python tools/check_reference.py ref.wav --ref-text "Текст" [--lang ru] [--write-trimmed ref_trim.wav]
Выводит рекомендованный --text для `vox voice create`/export_voice.py (если whisper нашёл расхождение).
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from vox.audio import load_wav, save_wav  # noqa: E402


def words(s):
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def edit_ops(ref, hyp):
    """Выравнивание слов (Левенштейн) → список (op, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], "")); i -= 1       # в тексте есть, в аудио нет
        else:
            ops.append(("ins", "", hyp[j - 1])); j -= 1       # в аудио есть, в тексте нет
    return ops[::-1], d[n, m] / max(1, n)


def transcribe(audio, sr, lang):
    a16 = audio
    if sr != 16000:
        t = np.arange(0, len(audio) / sr, 1 / 16000)
        a16 = np.interp(t, np.arange(len(audio)) / sr, audio).astype(np.float32)
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("small", compute_type="int8")
        segs, _ = m.transcribe(a16, language=lang, word_timestamps=True)
        out = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
        return out
    except ImportError:
        pass
    try:
        import whisper
        m = whisper.load_model("small")
        r = m.transcribe(a16, language=lang, word_timestamps=True, fp16=False)
        return [(w["word"].strip(), w["start"], w["end"]) for s in r["segments"] for w in s["words"]]
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("--ref-text", required=True)
    ap.add_argument("--lang", default="ru"); ap.add_argument("--write-trimmed", default="")
    a = ap.parse_args()
    audio, sr = load_wav(a.audio)
    audio = audio.astype(np.float32)
    dur = len(audio) / sr
    env = np.abs(audio); thr = 0.02 * max(env.max(), 1e-6)
    nz = np.nonzero(env > thr)[0]
    lead = nz[0] / sr if len(nz) else dur; tail = (len(audio) - 1 - nz[-1]) / sr if len(nz) else dur
    chars = len(re.sub(r"\s", "", a.ref_text))
    print(f"длительность {dur:.2f} с, знаков {chars}, темп {chars / dur:.1f} зн/с (обычно 11–17), тишина в начале {lead:.2f} с, в конце {tail:.2f} с")
    warn = []
    if not 8 <= chars / dur <= 20: warn.append("темп речи вне нормы → текст и аудио, скорее всего, не соответствуют друг другу")
    if tail < 0.1: warn.append("в конце нет паузы: аудио обрезано на звуке — хвост может «просочиться» в синтез; добавьте ~0.25 с тишины (--write-trimmed)")
    if dur < 3 or dur > 30: warn.append("референс короче 3 с или длиннее 30 с; лучше 5–15 с")
    ws = transcribe(audio, sr, a.lang)
    if ws is None:
        warn.append("whisper не установлен (pip install faster-whisper) — сверка текста с речью пропущена")
    else:
        hyp = [re.sub(r"[^\w]", "", w[0].lower()) for w in ws]
        ops, wer = edit_ops(words(a.ref_text), [h for h in hyp if h])
        print(f"WER текст↔речь: {wer * 100:.1f}%")
        for op, rw, hw in ops:
            if op == "del": print(f"  в тексте есть, в речи нет: «{rw}»")
            elif op == "ins": print(f"  в речи есть, в тексте нет: «{hw}»")
            elif op == "sub": print(f"  заменено: текст «{rw}» ← речь «{hw}»")
        if wer > 0.1: warn.append("текст не совпадает с речью (WER>10%): правьте --ref-text по расшифровке")
        if a.write_trimmed:
            # обрезаем по концу последнего слова, у которого есть пара в тексте
            want = set(words(a.ref_text)); last_end = None
            for (w, s, e), h in zip(ws, hyp):
                if h in want: last_end = e
            end = min(dur, (last_end if last_end is not None else dur) + 0.12)
            seg = audio[: int(end * sr)]
            seg = np.concatenate([seg, np.zeros(int(0.25 * sr), np.float32)])
            save_wav(a.write_trimmed, seg, sr)
            print(f"записано {a.write_trimmed}: {len(seg) / sr:.2f} с (обрезано по слову, +0.25 с тишины)")
    if a.write_trimmed and ws is None:
        end = min(dur, (nz[-1] / sr + 0.15) if len(nz) else dur)
        seg = np.concatenate([audio[: int(end * sr)], np.zeros(int(0.25 * sr), np.float32)])
        save_wav(a.write_trimmed, seg, sr)
        print(f"записано {a.write_trimmed}: обрезана тишина + 0.25 с паузы в конце (без whisper — по слову не режем)")
    print("\n".join("ВНИМАНИЕ: " + w for w in warn) if warn else "референс выглядит согласованным")


if __name__ == "__main__":
    main()
