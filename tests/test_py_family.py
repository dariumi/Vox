"""Семейство моделей, целиком написанное на Python (пересказ vox-test), даёт тот же звук, что родная C++-реализация.
Запуск: python3 tests/test_py_family.py"""
import math
import os
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)

import numpy as np

import vox
from vox.formats import read_tmod_manifest, write_tmod

TWO_PI = 2 * math.pi


class PyState:
    def __init__(self):
        self.text = []
        self.frame = 0
        self.code0 = 0
        self.phase = 0.0
        self.base_hz = 160.0


class PyTestBackend:
    def __init__(self, tmod: vox.Tmod):
        a = tmod.manifest["audio"]
        self.sr, self.spf = a["sample_rate"], a["samples_per_frame"]
        self.nc, self.cs = a["num_codebooks"], a["codebook_size"]
        self.weights = tmod.blob("test.weights")  # проверка доступа к весам из mmap
        assert self.weights.shape == (64, 64)

    def info(self):
        return dict(sample_rate=self.sr, samples_per_frame=self.spf, num_codebooks=self.nc, codebook_size=self.cs, eos_code=self.cs)

    def name(self):
        return "py-test"

    def fingerprint(self):
        return "py-test-1"

    def tokenize(self, text):
        return [ord(c) for c in text]

    def speaker_embedding(self, pcm, sr):
        e = np.zeros(16, dtype=np.float32)
        n = len(pcm)
        if n == 0:
            return e
        for s in range(16):
            a, b = n * s // 16, n * (s + 1) // 16
            seg = pcm[a:b].astype(np.float64)
            e[s] = math.sqrt(float((seg * seg).sum()) / max(1, b - a))
        mx = e.max()
        return e / mx if mx > 0 else e

    def encode_codes(self, pcm, sr):
        n = len(pcm)
        frames = max(1, math.ceil(n / sr * self.sr / self.spf))
        codes = np.zeros((frames, self.nc), dtype=np.int16)
        for f in range(frames):
            a, b = int(n * f / frames), int(n * (f + 1) / frames)
            acc = float(np.abs(pcm[a:b].astype(np.float64)).sum())
            base = int(acc * 1000.0 / max(1, b - a))
            for c in range(self.nc):
                codes[f, c] = (base + c * 7) % self.cs
        return codes

    def begin(self, voice, language):
        st = PyState()
        emb = voice["speaker_emb"]
        if len(emb):
            st.base_hz = 110.0 + 120.0 * float(np.sum(emb.astype(np.float64))) / len(emb)
        return st

    def append_text(self, st, tokens):
        st.text.extend(tokens)

    def needs_text(self, st, ended):
        return (not ended) and st.frame // 2 >= len(st.text)

    def talker_logits(self, st, ended):
        n = self.cs
        lg = np.zeros(n + 1, dtype=np.float32)
        for i in range(n):
            lg[i] = math.sin(i * 1.7 + st.frame * 0.31)
        t = st.frame // 2
        if ended and t >= len(st.text):
            lg[n] = 20.0
            return lg
        target = (st.text[t] * 31 + st.frame) % n
        lg[target] = 6.0
        lg[n] = -20.0
        return lg

    def accept_code0(self, st, code):
        st.code0 = code

    def predictor_logits(self, st, cb):
        n = self.cs
        lg = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lg[i] = math.cos(i * 0.9 + cb)
        lg[(st.code0 + cb * 13) % n] = 6.0
        return lg

    def end_frame(self, st):
        st.frame += 1

    def decode(self, st, codes):
        out = np.empty(len(codes) * self.spf, dtype=np.float32)
        k = 0
        for f in range(len(codes)):
            hz = st.base_hz + 3.0 * int(codes[f, 0])
            amp = 0.15 + 0.05 * (int(codes[f, 1 if self.nc > 1 else 0]) % 4) / 3.0
            for _ in range(self.spf):
                st.phase += TWO_PI * hz / self.sr
                if st.phase > TWO_PI:
                    st.phase -= TWO_PI
                out[k] = amp * math.sin(st.phase)
                k += 1
        return out


class PyTestFamily(vox.Family):
    name = "py-test"
    description = "vox-test re-implemented in Python"

    def footprint(self, manifest):
        return {"stages": [{"name": "model", "prefixes": ["test."], "session_ram_bytes": 1 << 16}]}

    def capabilities(self, manifest):
        return {"clone_icl": True, "clone_xvector": True, "languages": ["auto"]}

    def create(self, tmod, plan):
        assert plan["backend"] == "cpu"
        return PyTestBackend(tmod)


def ref_audio(sr=24000, secs=3):
    t = np.arange(sr * secs, dtype=np.float32)
    return (0.3 * np.sin(t * 0.05) * (t % 4000) / 4000).astype(np.float32), sr


def main():
    vox.register_family(PyTestFamily())
    with tempfile.TemporaryDirectory() as d:
        native = vox.make_test_model(os.path.join(d, "native.tmod"))
        # та же модель, но с именем семейства py-test (веса — те же блобы)
        man = read_tmod_manifest(native)
        man["model"]["family"] = "py-test"
        w = np.array([(i % 97) / 97.0 for i in range(4096)], dtype=np.float32).reshape(64, 64)
        pym = os.path.join(d, "py.tmod")
        write_tmod(pym, man, {"test.weights": w})

        assert vox.inspect(pym)["family"] == "py-test"
        assert vox.plan(pym, "cpu")["feasible"]
        ref = ref_audio()
        with vox.Engine(native, "cpu") as en, vox.Engine(pym, "cpu") as ep:
            assert ep.capabilities["clone_icl"] and ep.info["family"] == "py-test", ep.info
            vn = en.clone_voice(ref, ref_text="тест")
            vp = ep.clone_voice(ref, ref_text="тест")
            for text, seed in (("привет мир", 1), ("hello world, this is a longer sentence.", 5)):
                an = en.synthesize(text, vn, seed=seed)
                ap = ep.synthesize(text, vp, seed=seed)
                assert an.shape == ap.shape and an.size > 0, (an.shape, ap.shape)
                err = float(np.max(np.abs(an - ap)))
                assert err == 0.0, f"python family differs from C++: max err {err}"
                print(f"ok: {text!r} seed {seed}: {an.size} samples identical")
            # потоковая подача текста частями
            with ep.session(vp, seed=3) as s:
                out = []
                for piece in ["привет, ", "это ", "поток."]:
                    s.feed(piece)
                    out += list(s.ready())
                s.end()
                out += list(s)
            whole = ep.synthesize("привет, это поток.", vp, seed=3)
            assert np.array_equal(np.concatenate(out), whole), "chunked text input changes result"
            print("ok: streaming text input equals one-shot")
            # ошибка в Python-бэкенде приходит как VoxError, а не падение процесса
        class Broken(PyTestFamily):
            name = "py-broken"

            def create(self, tmod, plan):
                raise RuntimeError("boom")

        vox.register_family(Broken())
        man["model"]["family"] = "py-broken"
        bm = os.path.join(d, "b.tmod")
        write_tmod(bm, man, {"test.weights": w})
        try:
            vox.Engine(bm, "cpu")
            raise SystemExit("broken family must fail")
        except vox.VoxError as e:
            assert "boom" in str(e), e
            print("ok: exception in Python backend -> VoxError:", e)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
