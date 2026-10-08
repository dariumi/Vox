"""Тесты Python-API. Запуск: python -m pytest tests/test_python.py (нужен собранный пакет и ./vox)."""
import asyncio
import os
import subprocess
import sys
import tempfile

import numpy as np

import vox
from vox.formats import VoiceProfile, read_tmod_manifest, write_tmod

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.environ.get("VOX_CLI", os.path.join(HERE, "..", "build", "vox"))


def make_model(d):
    path = os.path.join(d, "test.tmod")
    subprocess.run([CLI, "make-test-model", "-o", path], check=True, capture_output=True)
    return path


def ref_audio(sr=24000, secs=3):
    t = np.arange(sr * secs, dtype=np.float32)
    return (0.3 * np.sin(t * 0.05) * (t % 4000) / 4000).astype(np.float32), sr


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        eng = vox.Engine(make_model(d))
        assert eng.sample_rate == 24000
        assert eng.info["backend"] == "null-cpu"

        pcm, sr = ref_audio()
        vox.save_wav(os.path.join(d, "ref.wav"), pcm, sr)
        voice = eng.clone_voice(os.path.join(d, "ref.wav"), "Текст референса", name="my_voice", seed=7)
        assert voice.name == "my_voice"
        vpath = os.path.join(d, "my_voice.tvoice")
        voice.save(vpath)

        # Профиль читается и чистым Python
        prof = VoiceProfile.load(vpath)
        assert prof.ref_codes.shape == (38, 4) and prof.sampling["seed"] == 7

        v2 = eng.load_voice(vpath)
        text = "Привет! Это проверка Python-обёртки."
        a = eng.synthesize(text, voice=v2, seed=42)
        b = eng.synthesize(text, voice=vpath, seed=42)   # голос можно передать путём
        c = eng.synthesize(text, voice=v2, seed=43)
        assert a.dtype == np.float32 and len(a) > 0
        assert np.array_equal(a, b) and not np.array_equal(a, c)

        # Стриминг текста частями == синтез целиком
        out = []
        with eng.session(v2, seed=42, chunk_samples=1000) as s:
            for piece in ["При", "вет! Это про", "верка Python-обёр", "тки."]:
                s.feed(piece)
                out.extend(s.ready())
            s.end()
            out.extend(s)
        assert np.array_equal(np.concatenate(out), a)

        async def collect():
            return [c async for c in eng.astream(text, voice=v2, seed=42)]
        assert np.array_equal(np.concatenate(asyncio.run(collect())), a)

        wav = os.path.join(d, "out.wav")
        vox.save_wav(wav, a, eng.sample_rate)
        back, sr2 = vox.load_wav(wav)
        assert sr2 == 24000 and np.max(np.abs(back - a)) < 1e-3


def test_errors():
    with tempfile.TemporaryDirectory() as d:
        model = make_model(d)
        try:
            vox.Engine(model, device="cuda:0")
            assert False, "expected VoxError"
        except vox.VoxError as e:
            assert e.status == -3 and "unknown device" in e.message
        eng = vox.Engine(model)
        pcm, sr = ref_audio()
        try:
            eng.clone_voice((pcm, sr))  # ICL без текста референса
            assert False
        except vox.VoxError as e:
            assert e.status == -6
        xv = eng.clone_voice((pcm, sr), xvector_only=True)
        assert xv.info["mode"] == "xvector_only"
        try:
            eng.synthesize("x", voice=xv, temprature=0.5)
            assert False
        except TypeError:
            pass


def test_formats_python():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.tmod")
        write_tmod(p, {"model": {"family": "vox-test", "weights_sha256": "x"},
                       "audio": {"num_codebooks": 4, "codebook_size": 64}},
                   {"test.weights": np.ones((8, 8), np.float32)})
        m = read_tmod_manifest(p)
        assert m["blobs"]["test.weights"]["shape"] == [8, 8]
        eng = vox.Engine(p)  # файл из Python открывается нативным рантаймом
        assert eng.info["num_codebooks"] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    sys.exit(0)
