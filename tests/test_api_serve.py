"""Python-API со стримингом и HTTP-сервер на крошечной модели. Нужен vocab gguf Qwen2 (VOX_QWEN2_GGUF), иначе SKIP."""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import vox
from vox.build import build
from vox.formats import VoiceProfile
GGUF = os.environ.get("VOX_QWEN2_GGUF", "")
if not GGUF or not os.path.exists(GGUF):
    print("SKIP: задайте VOX_QWEN2_GGUF=путь/к/ggml-vocab-qwen2.gguf"); sys.exit(0)
from test_qwen3_tiny import hf_tokenizer, weights_sha256
work = tempfile.mkdtemp(prefix="voxserve_")
try:  # with the reference implementation installed the tiny model also has the voice encoders (POST /v1/voices)
    import torch, qwen_tts  # noqa: F401,E401
    HAVE_REF = True
except ImportError:
    HAVE_REF = False
_gen = "make_tiny_official.py" if HAVE_REF else "make_tiny_model.py"
subprocess.run([sys.executable, os.path.join(ROOT, "tools", _gen), "--vocab", GGUF, "--out", work + "/model"], check=True, capture_output=True)
build(work + "/model", work + "/tiny_f32.tmod", dtype="f32")
_rng = np.random.default_rng(5); _rt = "Привет, это референс голоса."
_p = VoiceProfile(speaker_emb=_rng.normal(0, 1, 64).astype(np.float32), name="icl", mode="icl", ref_text=_rt,
                  model={"family": "qwen3-tts", "variant": "12hz-tiny-base", "weights_sha256": weights_sha256(work + "/model")})
_p.ref_codes = _rng.integers(0, 64, (10, 4)).astype(np.int16)
_p.ref_text_tokens = np.array(hf_tokenizer(GGUF).encode("<|im_start|>assistant\n" + _rt + "<|im_end|>\n", add_special_tokens=False).ids, np.int32)
_p.save(work + "/tiny_icl.tvoice")
# одно и то же устройство у библиотеки и сервера: с auto они выбирали разные видеокарты (выход совпадал только до ~3e-6)
DEV = ([d for d in os.environ.get("VOX_TEST_DEVICES", "cpu").split(";") if d] or ["cpu"])[0]
eng = vox.Engine(work + "/tiny_f32.tmod", device=DEV)
v = eng.load_voice(work + "/tiny_icl.tvoice")
text = "Hello, this is a streaming text test. It has several sentences, and some punctuation!"
a = eng.synthesize(text, voice=v, seed=7, language="english")
# текст кусками произвольной нарезки (в т.ч. посреди слов) → то же аудио
rng = np.random.default_rng(0)
for trial in range(5):
    cuts = sorted(rng.choice(np.arange(1, len(text)), size=rng.integers(2, 9), replace=False))
    pieces = [text[i:j] for i, j in zip([0] + list(cuts), list(cuts) + [len(text)])]
    out = []
    with eng.session(v, seed=7, language="english") as s:
        for p in pieces:
            s.feed(p); out.extend(s.ready())
        s.end(); out.extend(s)
    b = np.concatenate(out)
    assert np.array_equal(a, b), f"piecewise text differs (trial {trial}, pieces {pieces})"
print("piecewise streaming text == whole text: OK (5 random splits)", len(a), "samples")
# NFC: «й» как и + combining breve
t1 = "Привет, это тест."; t2 = t1.replace("и", "и\u0306") 
a1 = eng.synthesize("Привет, это тест. Й", voice=v, seed=1); a2 = eng.synthesize("Привет, это тест. И\u0306", voice=v, seed=1)
assert np.array_equal(a1, a2); print("NFC normalisation: OK")
# ошибка языка
try: eng.synthesize("x", voice=v, language="klingon"); print("no error?!")
except vox.VoxError as e: print("bad language ->", e.status, e.message)
# HTTP-сервер
os.makedirs(work + "/voices", exist_ok=True); import shutil; shutil.copy(work + "/tiny_icl.tvoice", work + "/voices/demo.tvoice")
# свободный порт (на занятом 18080 запросы уходили к серверу, оставшемуся от прошлого прогона, — отсюда случайное «stream differs»)
import atexit, socket, tempfile
with socket.socket() as _s:
    _s.bind(("127.0.0.1", 0)); PORT = _s.getsockname()[1]
_srv_log = tempfile.TemporaryFile()
proc = subprocess.Popen([sys.executable, "-m", "vox", "serve", "--model", work + "/tiny_f32.tmod", "--voices", work + "/voices", "--port", str(PORT), "--device", DEV],
                        stdout=_srv_log, stderr=_srv_log)
atexit.register(lambda: proc.poll() is None and (proc.kill(), proc.wait()))   # сервер гаснет и при упавшем assert
for _ in range(100):
    if proc.poll() is not None:
        _srv_log.seek(0); sys.exit("vox serve exited: " + _srv_log.read().decode(errors="replace")[-2000:])
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
    except Exception: time.sleep(0.2)
print("voices:", json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/voices")))
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/tts", data=json.dumps({"text": text, "voice": "demo", "seed": 7, "language": "english", "format": "f32"}).encode())
body = urllib.request.urlopen(req).read()
got = np.frombuffer(body, "<f4")
assert np.array_equal(got, a), "server f32 stream differs from library output"
# a profile with two voices: listed as "cast:a" / "cast:b", spoken like the library with voice_id
cast = vox.Profile("cast"); cast.add(vox.Voice.open(work + "/tiny_icl.tvoice"), as_voice="a"); cast.add(vox.Voice.open(work + "/tiny_icl.tvoice"), as_voice="b")
cast.set_voice("b", temperature=0.5); cast.save(work + "/voices/cast.tvoice")
lst = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/voices"))
assert "cast:a" in json.dumps(lst) and "cast:b" in json.dumps(lst), lst
body = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/tts", data=json.dumps({"text": text, "voice": "cast:b", "seed": 7, "language": "english", "format": "f32"}).encode())).read()
assert np.array_equal(np.frombuffer(body, "<f4"), eng.synthesize(text, vox.Voice.open(work + "/voices/cast.tvoice"), voice_id="b", seed=7, language="english")), "profile voice over HTTP differs"
print("profile voices over HTTP: ok")
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/tts", data=json.dumps({"text": text, "voice": "demo", "format": "wav"}).encode())
wav = urllib.request.urlopen(req).read(); print("server wav bytes:", len(wav), wav[:4])
try: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/tts", data=json.dumps({"text": "x", "voice": "nope"}).encode()))
except urllib.error.HTTPError as e: print("unknown voice ->", e.code)
# создание голоса через сервер (нативно) == Engine.clone_voice
import base64
if not HAVE_REF:
    proc.terminate(); print("server: OK (POST /v1/voices skipped: install torch + qwen-tts for the tiny encoder weights)"); sys.exit(0)
ref = os.path.join(work, "ref.wav")
_t = np.arange(24000 * 3) / 24000
from vox.audio import save_wav
save_wav(ref, (0.3 * np.sin(2 * np.pi * 220 * _t) * np.sin(2 * np.pi * 2 * _t)).astype(np.float32), 24000)
body = json.dumps({"name": "made", "audio_b64": base64.b64encode(open(ref, "rb").read()).decode(),
                   "ref_text": "Привет, это референс.", "language": "english"}).encode()
resp = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/voices", data=body))
assert resp.status == 201, resp.status
print("voices after create:", json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/voices")))
from vox.formats import VoiceProfile
a = VoiceProfile.load(os.path.join(work, "voices", "made.tvoice"))
b_ = eng.clone_voice(ref, "Привет, это референс.")
b_.save(os.path.join(work, "lib.tvoice")); b = VoiceProfile.load(os.path.join(work, "lib.tvoice"))
assert np.allclose(a.speaker_emb, b.speaker_emb, rtol=0, atol=1e-5 * max(1.0, float(np.abs(a.speaker_emb).max()))) \
    and np.array_equal(a.ref_codes, b.ref_codes), "server voice != library voice"
try: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/voices", data=json.dumps({"name": "../x", "audio_b64": ""}).encode()))
except urllib.error.HTTPError as e: print("bad voice name ->", e.code)
proc.terminate()
print("server: OK (f32 stream bit-identical to library; POST /v1/voices == Engine.clone_voice)")
