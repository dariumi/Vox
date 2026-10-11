"""Duplex streaming: text arrives in pieces (like tokens from an LLM) and audio is taken as it becomes ready.

    python examples/python/duplex_example.py model.tmod voice.tvoice out.wav

Writing happens in one thread, reading in the main one; commit="auto": the model decides how much text it needs
(Qwen3-TTS reads word by word and starts speaking at once, F5-TTS speaks finished chunks)."""
import sys
import threading
import time
import wave

import numpy as np
import vox

model, voice_path, out = sys.argv[1:4]
eng = vox.Engine(model)
print("the model takes text like this:", eng.capabilities.get("stream"))

pieces = ["Hello! ", "This is duplex ", "streaming: the text arrives ", "in small pieces, ", "and the audio starts at once."]
with eng.duplex(voice_path, commit="auto", idle_ms=600, seed=42) as d:
    def typist():                       # from anywhere: another thread, a coroutine, a socket
        for p in pieces:
            d.write(p)
            time.sleep(0.05)
        d.end()                         # the text has ended
    threading.Thread(target=typist, daemon=True).start()
    chunks = []
    for chunk in d.audio():             # blocking iterator: audio arrives while the text is still being written
        chunks.append(chunk)
        print(f"audio: +{len(chunk) / eng.sample_rate:.2f} s")
    print("seconds to first audio:", d.stats["first_audio_s"])

pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(eng.sample_rate)
    w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
print("written:", out, f"{len(pcm) / eng.sample_rate:.1f} s")
