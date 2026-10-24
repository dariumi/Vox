#!/usr/bin/env python3
"""Builds the project site into docs/ (GitHub Pages: Settings -> Pages -> Deploy from a branch -> /docs).
Sources are in site/ (index.tmpl.html is English, index.ru.tmpl.html is Russian, i18n.*.js, style.css, app.js, assets/); code samples are
taken straight from examples/ so the page shows exactly the code the tests check. vox.js is built by tools/build_web.py.
Output: docs/index.html (English) and docs/index.ru.html (Russian)."""
import html
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "build_web.py")])
S, D = os.path.join(ROOT, "site"), os.path.join(ROOT, "docs")


def read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read().rstrip("\n")



def build_langs(ru):
    t = lambda en, rus: rus if ru else en
    VERIFIED = t("verified: the audio matches the CLI bit for bit", "проверено: звук совпадает с CLI до бита")
    return [
    ("cli", "CLI", "", "vox say / vox voice create", t("""# a voice from a recording (best: 24 kHz WAV, 5-15 seconds of clean speech)
vox voice create ref.wav --model model.tmod --text "The text spoken in the recording" -o me.tvoice

# synthesize to a file
vox say "Hello! This is a check." --model model.tmod --voice me.tvoice -o out.wav

# stream into a player
vox say --stdin --stream --model model.tmod --voice me.tvoice | ffplay -f f32le -ar 24000 -i -

# F5-TTS: the parameters are named as in the official code
vox say "Hello!" --model f5.tmod --voice me.tvoice --nfe 16 --cfg 2 --sway -1 --speed 1 -o out.wav

# Kokoro-82M: a named voice or a blend, no recording needed
vox say "Hello!" --model kokoro.tmod --speaker af_heart,bf_emma -o out.wav
vox phonemize --model kokoro.tmod "It's 10 o'clock.\"""", """# голос из записи (лучше WAV 24 кГц, 5–15 секунд чистой речи)
vox voice create ref.wav --model model.tmod --text "Текст, который сказан в записи" -o me.tvoice

# синтез в файл
vox say "Привет! Это проверка." --model model.tmod --voice me.tvoice -o out.wav

# потоком в плеер
vox say --stdin --stream --model model.tmod --voice me.tvoice | ffplay -f f32le -ar 24000 -i -

# F5-TTS: параметры называются как в официальном коде
vox say "Привет!" --model f5.tmod --voice me.tvoice --nfe 16 --cfg 2 --sway -1 --speed 1 -o out.wav

# Kokoro-82M (английский): именной голос или смесь, без записи
vox say "Hello!" --model kokoro.tmod --speaker af_heart,bf_emma -o out.wav
vox phonemize --model kokoro.tmod "It's 10 o'clock.\""""),
     t("Device list and the placement plan without loading weights: <code>vox devices</code>, <code>vox plan --model model.tmod</code>.", "Список устройств и план без загрузки весов: <code>vox devices</code>, <code>vox plan --model model.tmod</code>.")),
    ("py", "Python", "", "import vox", t("""import vox

eng = vox.Engine("model.tmod", device="auto")          # "cpu", "vulkan:0", "auto"
voice = eng.clone_voice("ref.wav", "The text spoken in the recording", name="me", seed=42)
voice.save("me.tvoice")

audio = eng.synthesize("Hello!", voice, seed=42)        # numpy float32, mono, eng.sample_rate

# text arrives in pieces, audio is taken as it becomes ready
with eng.session(voice, seed=42) as s:
    s.feed("The first sentence. ")
    chunks = list(s.ready())
    s.end()
    chunks += list(s)""", """import vox

eng = vox.Engine("model.tmod", device="auto")          # "cpu", "vulkan:0", "auto"
voice = eng.clone_voice("ref.wav", "Текст, который сказан в записи", name="me", seed=42)
voice.save("me.tvoice")

audio = eng.synthesize("Привет!", voice, seed=42)       # numpy float32, mono, eng.sample_rate

# текст приходит по кускам, звук забирается по мере готовности
with eng.session(voice, seed=42) as s:
    s.feed("Первая фраза. ")
    chunks = list(s.ready())
    s.end()
    chunks += list(s)"""), t("Install: <code>pip install .</code> from the repository root. Errors are <code>vox.VoxError</code>, running out of memory is <code>vox.OutOfMemory</code>.", "Установка: <code>pip install .</code> из корня репозитория. Ошибки — <code>vox.VoxError</code>, нехватка памяти — <code>vox.OutOfMemory</code>.")),
    ("c", "C", VERIFIED, "examples/minimal.c", read("examples/minimal.c"), t("Build: <code>cc examples/minimal.c -Iinclude -Lbuild -lvox -o minimal</code>. The contract is <code>include/vox.h</code>.", "Сборка: <code>cc examples/minimal.c -Iinclude -Lbuild -lvox -o minimal</code>. Контракт — <code>include/vox.h</code>.")),
    ("cpp", "C++", VERIFIED, "examples/minimal.cpp", read("examples/minimal.cpp"), ""),
    ("rust", "Rust", VERIFIED, "examples/rust/src/main.rs", read("examples/rust/src/main.rs"), t("No dependencies. Set <code>VOX_LIB_DIR</code> to the library path at build time and <code>LD_LIBRARY_PATH</code> at run time.", "Без зависимостей. Путь к библиотеке — переменная <code>VOX_LIB_DIR</code> при сборке, при запуске нужен <code>LD_LIBRARY_PATH</code>.")),
    ("go", "Go", VERIFIED, "examples/go/main.go", read("examples/go/main.go"), t("Uses cgo, so a C compiler is required.", "cgo, поэтому нужен компилятор C.")),
    ("node", "Node.js", VERIFIED, "examples/node/vox.mjs", read("examples/node/vox.mjs"), t("Through the <code>koffi</code> FFI library (<code>npm install</code> in the example folder).", "Через FFI-библиотеку <code>koffi</code> (<code>npm install</code> в папке примера).")),
    ("ctypes", "Python ctypes", VERIFIED, "examples/python/ctypes_example.py", read("examples/python/ctypes_example.py"), t("If you do not want to build the Python module: plain <code>ctypes</code> over the same library.", "Если не хотите собирать Python-модуль: чистый <code>ctypes</code> поверх той же библиотеки.")),
    ("web", t("Browser", "Браузер"), t("verified in Chromium (WebGPU via SwiftShader) on test models", "проверено в Chromium (WebGPU через SwiftShader) на тестовых моделях"), "vox.js", t("""<script src="vox.js"></script>
<script type="module">
  // 1. The model: a Hugging Face repo, a direct link, a .tmod, a File/Blob. A checkpoint (safetensors) is downloaded and
  //    converted in the browser; Qwen3-TTS is read tensor by tensor (HTTP Range) and never sits in memory as a whole.
  //    backend: "auto" (default) uses WebGPU and falls back to the CPU; load: "balanced" | "gentle" | "full" paces the GPU.
  const model = await Vox.load("hf:Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", {
    dtype: "bf16", quant: "",             // "q8" is four times smaller than f32
    onProgress: (p) => console.log(Math.round(p * 100) + "%"), onStatus: console.log,
  });
  // F5-TTS the same way: Vox.load("hf:SWivid/F5-TTS") or Vox.load("https://.../model_1250000.safetensors", { vocab: "https://.../vocab.txt" })

  // 2. The voice. F5-TTS and Qwen3-TTS Base: from a recording (File, Blob, URL or Float32Array + sampleRate) and its text.
  //    Qwen3-TTS CustomVoice/VoiceDesign: a ready speaker / voice description, or a .tvoice (vox voice create or voice.toTvoice())
  const voice = model.family === "qwen3-tts" && !model.capabilities.clone_from_audio
    ? model.speakerVoice("aiden", "Speak slowly and calmly", { language: "english" })
    : await model.cloneVoice(refFile, "The text spoken in the recording");

  // 3. Synthesize in one piece or as a stream
  const { pcm, sampleRate } = await model.synthesize("Hello, world!", { voice, seed: 1 });
  Vox.play(pcm, sampleRate);            // or Vox.wav(pcm, sampleRate) -> a Blob to download

  // 4. Duplex streaming: text goes in on the fly, audio comes out on the fly
  const player = new Vox.StreamPlayer(model.sampleRate);
  const d = model.duplex({ voice, commit: "auto", idle_ms: 600 });   // auto | sentence | manual
  input.addEventListener("input", () => d.setText(input.value));     // a text field; d.write(token from an LLM) works too
  (async () => { for await (const piece of d.audio()) player.push(piece); })();
  // d.flush() speaks what has accumulated now; d.end() means the text is finished; d.cancel() aborts
</script>""", """<script src="vox.js"></script>
<script type="module">
  // 1. модель: что угодно — репозиторий HF, прямая ссылка, .tmod, File/Blob. Чекпойнт (safetensors) качается и
  //    конвертируется в браузере сам; Qwen3-TTS читается по тензорам (HTTP Range), целиком в память не попадает.
  //    backend: "auto" (по умолчанию) — WebGPU, иначе процессор; load: "balanced" | "gentle" | "full" — нагрузка на видеокарту.
  const model = await Vox.load("hf:Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", {
    dtype: "bf16", quant: "",             // "q8" — вчетверо меньше f32
    onProgress: (p) => console.log(Math.round(p * 100) + "%"), onStatus: console.log,
  });
  // F5-TTS так же: Vox.load("hf:SWivid/F5-TTS") или Vox.load("https://…/model_1250000.safetensors", { vocab: "https://…/vocab.txt" })

  // 2. голос. F5-TTS и Qwen3-TTS Base: из записи (File, Blob, URL или Float32Array + sampleRate) и текста записи.
  //    Qwen3-TTS CustomVoice/VoiceDesign: готовый диктор / описание голоса или .tvoice (vox voice create или voice.toTvoice())
  const voice = model.family === "qwen3-tts" && !model.capabilities.clone_from_audio
    ? model.speakerVoice("aiden", "Speak slowly and calmly", { language: "english" })
    : await model.cloneVoice(refFile, "Текст, который сказан в записи");

  // 3. синтез целиком или потоком
  const { pcm, sampleRate } = await model.synthesize("Привет, мир!", { voice, seed: 1 });
  Vox.play(pcm, sampleRate);            // или Vox.wav(pcm, sampleRate) → Blob для скачивания

  // 4. двусторонний стриминг: текст подаётся на лету, звук забирается на лету
  const player = new Vox.StreamPlayer(model.sampleRate);
  const d = model.duplex({ voice, commit: "auto", idle_ms: 600 });   // auto | sentence | manual
  input.addEventListener("input", () => d.setText(input.value));     // поле ввода; можно и d.write(токен из LLM)
  (async () => { for await (const piece of d.audio()) player.push(piece); })();
  // d.flush() — озвучить накопленное сейчас; d.end() — текст закончился; d.cancel() — оборвать
</script>"""), t("Qwen3-TTS reads text word by word and can speak mid-sentence, F5-TTS speaks whole chunks (<code>d.caps</code> shows how the model takes text). F5 parameters are named as in the official code: <code>nfe_step</code>, <code>cfg_strength</code>, <code>sway_sampling_coef</code>, <code>speed</code>. Cancel with <code>signal</code> (AbortSignal).", "Qwen3-TTS читает текст по словам и говорит на полуслове, F5-TTS озвучивает целыми кусками (<code>d.caps</code> показывает, как модель принимает текст). Параметры F5 называются как в официальном коде: <code>nfe_step</code>, <code>cfg_strength</code>, <code>sway_sampling_coef</code>, <code>speed</code>. Отмена — <code>signal</code> (AbortSignal).")),
    ("http", "HTTP", "", "python -m vox serve", t("""python -m vox serve --model model.tmod --voices ./voices --port 8080 --device vulkan

curl localhost:8080/v1/voices
curl -X POST localhost:8080/v1/tts -d '{"text":"Hello!","voice":"me","seed":42}' | ffplay -i -
curl -X POST localhost:8080/v1/voices -d '{"name":"me","audio_b64":"<WAV in base64>","ref_text":"The recording text"}'""", """python -m vox serve --model model.tmod --voices ./voices --port 8080 --device vulkan

curl localhost:8080/v1/voices
curl -X POST localhost:8080/v1/tts -d '{"text":"Привет!","voice":"me","seed":42}' | ffplay -i -
curl -X POST localhost:8080/v1/voices -d '{"name":"me","audio_b64":"<WAV в base64>","ref_text":"Текст записи"}'"""),
     t("Audio is streamed as it becomes ready. Response formats: <code>wav</code>, <code>pcm16</code>, <code>f32</code>.", "Звук отдаётся потоком по мере готовности. Форматы ответа: <code>wav</code>, <code>pcm16</code>, <code>f32</code>.")),
    ]


ver = re.search(r"project\(vox VERSION ([0-9.]+)", read("CMakeLists.txt")).group(1)
os.makedirs(D, exist_ok=True)
for ru, tmpl, out in ((False, "site/index.tmpl.html", "index.html"), (True, "site/index.ru.tmpl.html", "index.ru.html")):
    buttons, panels = [], []
    COPY = "Копировать" if ru else "Copy"
    for i, (lid, label, tag, src, code, note) in enumerate(build_langs(ru)):
        sel = "true" if i == 0 else "false"
        buttons.append(f'<button role="tab" id="tab-{lid}" aria-controls="panel-{lid}" aria-selected="{sel}" tabindex="{0 if i == 0 else -1}">{html.escape(label)}</button>')
        meta = f'<span>{html.escape(src)}</span>' + (f'<span class="tag">{html.escape(tag)}</span>' if tag else "") + f'<button class="copy" data-target="code-{lid}" type="button">{COPY}</button>'
        panels.append(
            f'<div class="tabpanel" role="tabpanel" id="panel-{lid}" aria-labelledby="tab-{lid}"{"" if i == 0 else " hidden"}>'
            f'<div class="meta">{meta}</div><pre id="code-{lid}"><code>{html.escape(code)}</code></pre>'
            + (f'<div class="note">{note}</div>' if note else "") + "</div>")
    t = read(tmpl).replace("{{LANG_BUTTONS}}", "".join(buttons)).replace("{{LANG_PANELS}}", "\n".join(panels)).replace("{{VERSION}}", ver)
    open(os.path.join(D, out), "w", encoding="utf-8").write(t + "\n")
for f in ("style.css", "app.js", "i18n.en.js", "i18n.ru.js"):
    shutil.copy(os.path.join(S, f), D)
for f in ("vox.js", "vox.mjs", "pinyin.json"):   # pinyin.json is only needed for Chinese text in F5-TTS; it sits next to vox.js
    src = os.path.join(ROOT, "webgpu", "web", "dist", f)
    if os.path.exists(src):
        shutil.copy(src, D)
shutil.copytree(os.path.join(S, "assets"), os.path.join(D, "assets"), dirs_exist_ok=True)
open(os.path.join(D, ".nojekyll"), "w").close()
print("site: docs/index.html + docs/index.ru.html, vox %s" % ver)
