"""Сквозная проверка бэкенда Qwen3-TTS на крошечной модели со случайными весами против независимой
numpy-реализации (tools/ref_numpy.py). Запуск: python tests/test_qwen3_tiny.py [build/vox]
Нужны: numpy, tokenizers, regex и vocab-only gguf Qwen2 (llama.cpp/models/ggml-vocab-qwen2.gguf)."""
import os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "python"), HERE, os.path.join(ROOT, "tools")]
from vox.formats import VoiceProfile
from vox.build import weights_sha256, build
from test_tokenizer import hf_tokenizer

GGUF = os.environ.get("VOX_QWEN2_GGUF", "qwen2.gguf")
CASES = [  # (voice, text, language)
    ("icl", "Hi.", "english"),
    ("icl", "Hello, this is a test.", "english"),
    ("icl", "Привет! Это длинная проверка движка синтеза речи, слово за словом.", "russian"),
    ("icl", "Auto language  with   odd spacing,and punctuation!?", "auto"),
    ("xv", "Hello world", "english"),
    ("xv", " Привет, как дела у тебя сегодня? Всё хорошо.", "auto"),
]

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr

def main(cli):
    if not os.path.exists(GGUF):
        print("SKIP: set VOX_QWEN2_GGUF=path/to/ggml-vocab-qwen2.gguf")
        return 0
    work = tempfile.mkdtemp(prefix="voxtiny_")
    model_dir = os.path.join(work, "model")
    assert run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_model.py"), "--vocab", GGUF, "--out", model_dir])[0] == 0
    sha = weights_sha256(model_dir)
    hf = hf_tokenizer(GGUF)
    rng = np.random.default_rng(5)
    voices = {}
    for kind in ("icl", "xv"):
        ref_text = "Привет, это референс голоса." if kind == "icl" else ""
        p = VoiceProfile(speaker_emb=rng.normal(0, 1, 64).astype(np.float32), name=kind,
                         mode="icl" if kind == "icl" else "xvector_only", ref_text=ref_text,
                         model={"family": "qwen3-tts", "variant": "12hz-tiny-base", "weights_sha256": sha})
        if kind == "icl":
            p.ref_codes = rng.integers(0, 76, (10, 4)).astype(np.int16)
            p.ref_text_tokens = np.array(hf.encode("<|im_start|>assistant\n" + ref_text + "<|im_end|>\n", add_special_tokens=False).ids, np.int32)
        voices[kind] = os.path.join(work, kind + ".tvoice")
        p.save(voices[kind])

    tmods = {}
    for name, kw in (("f32", dict(dtype="f32")), ("f16", dict(dtype="f16")), ("bf16", dict(dtype="bf16")), ("q8", dict(dtype="f32", quant="q8"))):
        tmods[name] = os.path.join(work, f"tiny_{name}.tmod")
        build(model_dir, tmods[name], **kw)

    failures = 0
    for i, (vk, text, lang) in enumerate(CASES):
        dump = os.path.join(work, f"case{i}.safetensors")
        rc, out = run([sys.executable, os.path.join(ROOT, "tools", "ref_numpy.py"), "--model", model_dir, "--voice", voices[vk],
                       "--text", text, "--lang", lang, "--out", dump])
        assert rc == 0, out
        for fmt, tol in (("f32", "2e-3"), ("f16", "2e-2"), ("bf16", "5e-2"), ("q8", "5e-2")):
            if fmt != "f32" and i not in (1, 4):  # квантованные форматы — на двух случаях (ICL и X-vector)
                continue
            # у не-f32 допускается расхождение жадного выбора, поэтому свободный полёт проверяем только для f32
            rc, out = run([cli, "verify", "--device", os.environ.get("VOX_TEST_DEVICE", "auto"), "--model", tmods[fmt], "--dump", dump, "--voice", voices[vk], "--tol", tol,
                           "--free", "1" if fmt == "f32" else "0"] + ([] if fmt == "f32" else ["--quant", "1"] if fmt == "q8" else ["--no-argmax"]))
            status = "PASS" if rc == 0 else "FAIL"
            print(f"case {i} [{vk}/{lang}] {fmt}: {status}")
            if rc != 0:
                failures += 1
                print(out)
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return failures

if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")) else 0)
