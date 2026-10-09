"""Двусторонний стриминг: текст приходит кусками (как токены из LLM), звук забирается по мере готовности.

    python examples/python/duplex_example.py model.tmod voice.tvoice out.wav

Запись в одном потоке, чтение — в основном; commit="auto": модель сама решает, сколько текста ей нужно
(Qwen3-TTS читает по словам и начинает говорить сразу, F5-TTS озвучивает законченные куски)."""
import sys
import threading
import time
import wave

import numpy as np
import vox

model, voice_path, out = sys.argv[1:4]
eng = vox.Engine(model)
print("модель принимает текст так:", eng.capabilities.get("stream"))

pieces = ["Здравствуйте! ", "Это двусторонний ", "стриминг: текст приходит ", "по кусочкам, ", "а звук идёт сразу."]
with eng.duplex(voice_path, commit="auto", idle_ms=600, seed=42) as d:
    def typist():                       # откуда угодно: другой поток, корутина, сокет
        for p in pieces:
            d.write(p)
            time.sleep(0.05)
        d.end()                         # текст закончился
    threading.Thread(target=typist, daemon=True).start()
    chunks = []
    for chunk in d.audio():             # блокирующий итератор: звук приходит, пока текст ещё пишется
        chunks.append(chunk)
        print(f"звук: +{len(chunk) / eng.sample_rate:.2f} с")
    print("до первого звука, с:", d.stats["first_audio_s"])

pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(eng.sample_rate)
    w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
print("записано:", out, f"{len(pcm) / eng.sample_rate:.1f} с")
