#!/usr/bin/env python3
"""Собирает сайт проекта в docs/ (GitHub Pages: Settings → Pages → Deploy from a branch → /docs).
Источники — site/ (index.tmpl.html, style.css, app.js, assets/); примеры кода подставляются из examples/ прямо из файлов,
чтобы на странице был ровно тот код, который проверяют тесты. vox.js собирается tools/build_web.py."""
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


VERIFIED = "проверено: звук совпадает с CLI до бита"
LANGS = [
    ("cli", "CLI", "", "vox say / vox voice create", """# голос из записи (лучше WAV 24 кГц, 5–15 секунд чистой речи)
vox voice create ref.wav --model model.tmod --text "Текст, который сказан в записи" -o me.tvoice

# синтез в файл
vox say "Привет! Это проверка." --model model.tmod --voice me.tvoice -o out.wav

# потоком в плеер
vox say --stdin --stream --model model.tmod --voice me.tvoice | ffplay -f f32le -ar 24000 -i -

# F5-TTS: параметры называются как в официальном коде
vox say "Привет!" --model f5.tmod --voice me.tvoice --nfe 16 --cfg 2 --sway -1 --speed 1 -o out.wav""",
     "Список устройств и план без загрузки весов: <code>vox devices</code>, <code>vox plan --model model.tmod</code>."),
    ("py", "Python", "", "import vox", """import vox

eng = vox.Engine("model.tmod", device="auto")          # "cpu", "vulkan:0", "auto"
voice = eng.clone_voice("ref.wav", "Текст, который сказан в записи", name="me", seed=42)
voice.save("me.tvoice")

audio = eng.synthesize("Привет!", voice, seed=42)       # numpy float32, mono, eng.sample_rate

# текст приходит по кускам, звук забирается по мере готовности
with eng.session(voice, seed=42) as s:
    s.feed("Первая фраза. ")
    chunks = list(s.ready())
    s.end()
    chunks += list(s)""", "Установка: <code>pip install .</code> из корня репозитория. Ошибки — <code>vox.VoxError</code>, нехватка памяти — <code>vox.OutOfMemory</code>."),
    ("c", "C", VERIFIED, "examples/minimal.c", read("examples/minimal.c"), "Сборка: <code>cc examples/minimal.c -Iinclude -Lbuild -lvox -o minimal</code>. Контракт — <code>include/vox.h</code>."),
    ("cpp", "C++", VERIFIED, "examples/minimal.cpp", read("examples/minimal.cpp"), ""),
    ("rust", "Rust", VERIFIED, "examples/rust/src/main.rs", read("examples/rust/src/main.rs"), "Без зависимостей. Путь к библиотеке — переменная <code>VOX_LIB_DIR</code> при сборке, при запуске нужен <code>LD_LIBRARY_PATH</code>."),
    ("go", "Go", VERIFIED, "examples/go/main.go", read("examples/go/main.go"), "cgo, поэтому нужен компилятор C."),
    ("node", "Node.js", VERIFIED, "examples/node/vox.mjs", read("examples/node/vox.mjs"), "Через FFI-библиотеку <code>koffi</code> (<code>npm install</code> в папке примера)."),
    ("ctypes", "Python ctypes", VERIFIED, "examples/python/ctypes_example.py", read("examples/python/ctypes_example.py"), "Если не хотите собирать Python-модуль: чистый <code>ctypes</code> поверх той же библиотеки."),
    ("web", "Браузер", "проверено в Chromium (WebGPU через SwiftShader) на тестовой модели", "vox.js", """<script src="vox.js"></script>
<script type="module">
  // 1. модель: URL, File, Blob или ArrayBuffer; onProgress — для индикатора загрузки
  const model = await Vox.load("f5.bf16.tmod", { onProgress: (p) => console.log(Math.round(p * 100) + "%") });

  // 2. голос: File из <input type=file>, Blob из MediaRecorder, URL или Float32Array (+ sampleRate)
  const voice = await model.cloneVoice(refFile, "Текст, который сказан в записи");
  voice.export();                       // JSON: сохраните и позже восстановите через Vox.Voice.from(json)

  // 3. синтез целиком
  const { pcm, sampleRate, stats } = await model.synthesize("Привет, мир!", {
    voice, nfe_step: 16, speed: 1, seed: 1,
    onStep: ({ chunk, chunks, step, steps }) => console.log(chunk + 1, "/", chunks, step, "/", steps),
  });
  Vox.play(pcm, sampleRate);            // или Vox.wav(pcm, sampleRate) → Blob для скачивания

  // 4. или потоком: звук играет, пока считаются следующие куски текста
  const player = new Vox.StreamPlayer(model.sampleRate);
  for await (const piece of model.stream("Длинный текст из нескольких предложений.", { voice })) player.push(piece);
</script>""", "Параметры называются как в официальном F5-TTS: <code>nfe_step</code>, <code>cfg_strength</code>, <code>sway_sampling_coef</code>, <code>speed</code>, <code>cross_fade_duration</code>. Отмена — <code>signal</code> (AbortSignal)."),
    ("http", "HTTP", "", "python -m vox serve", """python -m vox serve --model model.tmod --voices ./voices --port 8080 --device vulkan

curl localhost:8080/v1/voices
curl -X POST localhost:8080/v1/tts -d '{"text":"Привет!","voice":"me","seed":42}' | ffplay -i -
curl -X POST localhost:8080/v1/voices -d '{"name":"me","audio_b64":"<WAV в base64>","ref_text":"Текст записи"}'""",
     "Звук отдаётся потоком по мере готовности. Форматы ответа: <code>wav</code>, <code>pcm16</code>, <code>f32</code>."),
]

buttons, panels = [], []
for i, (lid, label, tag, src, code, note) in enumerate(LANGS):
    sel = "true" if i == 0 else "false"
    buttons.append(f'<button role="tab" id="tab-{lid}" aria-controls="panel-{lid}" aria-selected="{sel}" tabindex="{0 if i == 0 else -1}">{html.escape(label)}</button>')
    meta = f'<span>{html.escape(src)}</span>' + (f'<span class="tag">{html.escape(tag)}</span>' if tag else "") + f'<button class="copy" data-target="code-{lid}" type="button">Копировать</button>'
    panels.append(
        f'<div class="tabpanel" role="tabpanel" id="panel-{lid}" aria-labelledby="tab-{lid}"{"" if i == 0 else " hidden"}>'
        f'<div class="meta">{meta}</div><pre id="code-{lid}"><code>{html.escape(code)}</code></pre>'
        + (f'<div class="note">{note}</div>' if note else "") + "</div>")

ver = re.search(r"project\(vox VERSION ([0-9.]+)", read("CMakeLists.txt")).group(1)
t = read("site/index.tmpl.html").replace("{{LANG_BUTTONS}}", "".join(buttons)).replace("{{LANG_PANELS}}", "\n".join(panels)).replace("{{VERSION}}", ver)
os.makedirs(D, exist_ok=True)
open(os.path.join(D, "index.html"), "w", encoding="utf-8").write(t + "\n")
for f in ("style.css", "app.js"):
    shutil.copy(os.path.join(S, f), D)
for f in ("vox.js", "vox.mjs"):
    shutil.copy(os.path.join(ROOT, "webgpu", "web", "dist", f), D)
shutil.copytree(os.path.join(S, "assets"), os.path.join(D, "assets"), dirs_exist_ok=True)
open(os.path.join(D, ".nojekyll"), "w").close()
print("site: docs/index.html (%d KiB), vox %s" % (len(t.encode()) // 1024, ver))
