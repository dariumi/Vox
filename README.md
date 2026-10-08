# vox

Движок синтеза речи на C++17: C ABI (`include/vox.h`), CLI `vox`, Python-**библиотека** `vox` и HTTP-сервер со
стримингом. Без PyTorch, без ggml, без внешних библиотек: веса читаются через mmap из одного файла `.tmod`, голос
хранится в `.tvoice`. Бэкенды: CPU (x86-64 AVX2, aarch64), Vulkan (NVIDIA / AMD / Intel / Android) и WebGPU в браузере (`vox.js`, пока для F5-TTS). Форматы весов:
f32 / f16 / bf16 / q8 / q4. Встроенные семейства моделей: **Qwen3-TTS-12Hz** (Base: клонирование голоса;
CustomVoice: готовые дикторы и инструкции о стиле; VoiceDesign: голос по описанию; 0.6B и 1.7B) и **F5-TTS**
(DiT + flow matching + Vocos, клонирование по образцу и его тексту; не авторегрессионная). Другие модели
подключаются как «семейство» (C++ или Python), рантайм остаётся тем же. Очередь: XTTS, CosyVoice, Chatterbox.

- **Ресурсы под контролем**: на чём считать, сколько RAM и VRAM занять, сколько потоков и какого приоритета,
  режим работы (speed / balanced / memory / background), лимит одновременных сессий. План размещения можно посмотреть,
  ничего не загружая (`vox plan`, `vox.plan()`).

- **Весь пайплайн нативно**: создание голоса (мел-спектр, ECAPA-TDNN, энкодер Mimi с RVQ), talker, code predictor,
  потоковый декодер кодека. torch нужен только для эталонных проверок.
- **Совпадает с официальной реализацией** (`qwen-tts` 0.1.1): см. «Что проверено».
- **Детерминированно**: сэмплинг на счётчиковом RNG Philox. Результат зависит только от seed и не зависит от числа
  потоков и размера чанка.
- **Стриминг**: текст подаётся кусками, аудио забирается чанками по 4 кадра (320 мс). Состояние декодера
  переносится между чанками, швов нет.

> Статус: корректность проверена против официального кода, в том числе на настоящей 1.7B-Base. Замер 0.7 у
> пользователя (Ryzen 7 5700X, RTX 3050 6 ГБ, ~13 с речи): Vulkan q8 RTF 0.47, bf16 0.62, CPU q8 1.82; qwentts.cpp:
> CUDA 0.428, CPU 2.07. **Скорость 0.8 на железе не снимали**: новое (q4, гибридный talker, режимы ресурсов) проверено
> на корректность, а цифры скорости даст `tools/run_full_check.py` на вашей машине. Подробности и границы — в CHANGELOG
> и «Ограничениях».

## Сборка

    cmake -S . -B build -DVOX_NATIVE=ON && cmake --build build -j
    pip install .                     # Python-модуль (nanobind, scikit-build-core, numpy)

Vulkan включается сам, если найдены заголовки (`libvulkan-dev`). SPIR-V шейдеры лежат в репозитории
(`src/vk_shaders.inc`), glslang нужен только при их изменении (`python tools/gen_shaders.py`).
Termux: `pkg install clang cmake python python-numpy`, дальше так же. Сборка под aarch64 проверяется в qemu
(тесты и сверка с эталоном), на живом телефоне не запускалась.

## Быстрый старт

    # 1. модель Hugging Face → .tmod (q8 — самый быстрый на CPU; bf16 — без квантования)
    python -m vox build --model ./Qwen3-TTS-12Hz-1.7B-Base --dtype bf16 --quant q8 -o model.q8.tmod

    # 2. голос из записи — нативно, без torch (лучше WAV 24 кГц, 5–15 с чистой речи)
    vox voice create ref.wav --model model.q8.tmod --text "Текст, который звучит в записи" \
        --name my_voice --lang russian --seed 42 -o voices/my_voice.tvoice

    # 3. синтез
    vox say "Привет! Это проверка." --model model.q8.tmod --voice voices/my_voice.tvoice -o out.wav
    vox say --stdin --stream --model model.q8.tmod --voice voices/my_voice.tvoice | ffplay -f f32le -ar 24000 -i -

    # 4. сервер (стриминг WAV/PCM, создание голосов через POST /v1/voices)
    python -m vox serve --model model.q8.tmod --voices ./voices --port 8080 --device vulkan

Python:

```python
import vox
eng = vox.Engine("model.q8.tmod", device="vulkan:0")
voice = eng.clone_voice("ref.wav", "Текст референса", name="my_voice", seed=42)
voice.save("my_voice.tvoice")
audio = eng.synthesize("Привет!", voice)                 # numpy float32, 24 кГц
with eng.session(voice, seed=42) as s:                    # текст по кускам → аудио по мере готовности
    s.feed("Первая фраза. "); chunks = list(s.ready())
    s.end(); chunks += list(s)
```

### Python как библиотека (без CLI)

```python
import vox

vox.devices()                                   # CPU, RAM (с учётом лимита контейнера), GPU и их свободная память
vox.plan("model.q8.tmod", vox.Resources(device="gpu:1", vram_limit="3GiB"))   # куда что ляжет, веса не читаются
vox.inspect("model.q8.tmod")                    # семейство, вариант, стадии, возможности

res = vox.Resources(device="gpu:1", mode="balanced", ram_limit="6GiB", vram_limit="3GiB", threads=6,
                    cpu_affinity="0-5", priority="low", place={"codec": "cpu"}, max_sessions=2)
with vox.Engine("model.q8.tmod", resources=res) as eng:
    eng.plan, eng.capabilities, eng.speakers, eng.languages, eng.stats()
    voice = eng.clone_voice("ref.wav", "Текст референса")            # Base
    voice = eng.speaker_voice("Aiden", instruct="Speak slowly")      # CustomVoice (1.7B: с инструкцией)
    voice = eng.design_voice("A deep calm male voice")               # VoiceDesign
    wavs = eng.synthesize_many(["Первый.", "Второй."], voice, seed=1)   # параллельные сессии
    eng.benchmark("Текст замера.", voice, runs=3)                    # RTF, первый чанк, статистика — без CLI
```

Ошибки — `vox.VoxError`; нехватка памяти или лимит сессий — `vox.OutOfMemory`, нет устройства — `vox.DeviceError`.
Свою модель можно написать целиком на Python и зарегистрировать (`vox.register_family`): см.
[docs/python-backends.md](docs/python-backends.md).

Проверить референс до создания голоса: `python tools/check_reference.py ref.wav --ref-text "..."`. Скрипт смотрит
темп, паузу в конце, а с whisper ещё и расхождение текста с речью. Лишнее слово в начале синтеза почти всегда
означает, что текст референса не совпадает с записью.

## Ресурсы: устройство, память, режим

    vox devices [--json]                          # CPU, RAM, GPU с индексами и свободной видеопамятью
    vox plan --model m.tmod --mode memory --ram-limit 3GiB --vram-limit 2GiB     # план без загрузки весов
    vox say "…" --model m.tmod --voice v.tvoice --device gpu:1 --mode balanced --threads 6 --affinity 0-5 \
        --priority low --ram-limit 6GiB --vram-limit 3GiB --place codec=cpu --gpu-layers talker=14 --max-sessions 2
    --config resources.json                       # то же JSON-файлом; флаги командной строки его перекрывают

- **Планировщик** (`src/resources.cpp`) по описанию модели (стадии talker / predictor / codec: вес, трафик на кадр),
  системе (ядра, RAM с учётом cgroup, свободная память каждой видеокарты) и пожеланиям строит план: устройство,
  размещение каждой стадии, потоки, оценку RAM и VRAM. Сначала на GPU идут стадии с наибольшей пользой на байт.
  Если talker целиком не влезает, он **делится по слоям**: первые k слоёв и их KV-кэш на GPU, остальные на CPU.
  Свободная видеопамять берётся из `VK_EXT_memory_budget` (если расширения нет — оценка). Программные Vulkan-рендереры
  (llvmpipe) в `auto` не выбираются.
- **Режимы**: `speed` — все ядра, активное ожидание потоков; `balanced` (по умолчанию) — ядро остаётся ОС;
  `memory` — вдвое меньше потоков, без активного ожидания; `background` — четверть ядер, низкий приоритет, на GPU не идёт.
- **Лимиты — жёсткие**: превышение `ram_limit` / `vram_limit` в плане — отказ открыть модель (`OutOfMemory` с планом в
  тексте ошибки), если не задан `enforce=false`. Сессия, которая не влезает в остаток лимита (KV-кэш и состояние
  декодера), не создаётся; `max_sessions` ограничивает их число. После загрузки на GPU страницы mmap с весами
  возвращаются системе (`release_host_weights`, `--no-release` выключает).
- Потоки CPU у каждого движка свои (пул с привязкой к ядрам, `nice`): два движка в одном процессе с разным числом потоков
  дают одинаковый результат (проверено тестом).
- Ограничение: один Vulkan-девайс на процесс (несколько видеокарт в одном процессе не делятся).

### Устройства

    vox devices                                   # Vulkan-устройства с индексами
    --device cpu | auto | gpu[:N] | vulkan | vulkan:N
    --device vulkan:1,strict                      # ошибка, если хоть одна матрица не влезла в VRAM
    --device vulkan:1,codec=cpu                   # декодер кодека на CPU (параллельно с GPU), остальное на GPU

- Нумерация Vulkan не совпадает с nvidia-smi и CUDA, сверяйте устройства по имени из `vox devices`.
- На старте vox пишет в stderr, сколько матриц ушло в VRAM, сколько осталось на CPU и почему (`VOX_VERBOSE=1`
  печатает это всегда). Тихого отката нет; со `strict` (или `VOX_GPU_STRICT=1`) нехватка видеопамяти становится ошибкой.
- **На GPU**: talker и predictor целиком (нормы, RoPE, внимание с GQA, MLP, входная проекция и голова), веса слоя
  уложены слитно (q|k|v и gate|up). Talker — одна отправка на шаг. **Весь цикл predictor за кадр (15 проходов) — тоже
  одна отправка**: сэмплинг идёт в шейдере (тот же алгоритм и Philox, что на CPU), эмбеддинг выбранного кода берётся
  на GPU. Итого 2 синхронизации с хостом на кадр (было 16). Префилл с m > 8 токенов идёт тайловым GEMM, а не GEMV.
  Отключить сэмплинг на GPU: `VOX_GPU_SAMPLING=0`. **Декодер кодека тоже целиком на GPU**:
  свёрточный implicit-GEMM, Snake, ConvNeXt, транспонированные свёртки с overlap-add, трансформер со скользящим
  окном на кольцевом KV. Один командный буфер на чанк, наружу уходит только PCM.
- `codec=cpu` оставляет декодер на CPU. Тогда чанк k декодируется в отдельном потоке, пока GPU считает чанк k+1
  (отключить: `VOX_NO_OVERLAP=1`). Какой режим быстрее на вашей карте, покажет `bench` (ниже).

## Режимы декодера

- `--decoder stream` (по умолчанию): непрерывное потоковое декодирование, прогрев последними 50 кадрами референса
  (`--context N|all`). Минимальная задержка, швов нет даже на длинных фразах.
- `--decoder reference`: побитово та же схема, что в `qwen_tts` (`chunked_decode`: куски по 300 кадров, 25 кадров
  контекста, прогрев от начала куска). Нужна для сверки и A/B-сравнений с PyTorch и qwentts.cpp.

Разница между режимами — не погрешность. Официальный код с 300-го кадра «референс + речь» декодирует заново,
с коротким контекстом, поэтому звук после границы немного другой. Именно это давало ошибку 5.756e-2 в отчёте по 0.4.

## Что проверено

`tests/test_official.py` строит **официальными классами `qwen_tts`** крошечную модель со случайными весами
(`tools/make_tiny_official.py`: те же модули, имена тензоров и шаги кадра, что у 1.7B, только узкие) и сравнивает vox
с её `generate_voice_clone` и `create_voice_clone_prompt`:

| что | результат (CPU, Vulkan с декодером на GPU и на CPU) |
|---|---|
| эмбеддинг спикера (нативно) | отн. ошибка ≤ 4e-7 |
| коды референса (нативно, Mimi + RVQ) | 100% совпадение, в т.ч. референс 25 с (> 300 кадров) |
| логиты talker / predictor, f32 | ≤ 1e-6 / ≤ 1e-5, argmax 100% |
| аудио декодера, f32 | ≤ 3e-4 (официальный f32 сам отличается от своего же f64 на ~1e-4) |
| граница куска декодера внутри речи и внутри референса | совпадает (режим `reference`) |
| f16 / bf16 / q8 | допуски формата; q8 по KL и top-5 вместо argmax |
| aarch64 (qemu) | те же результаты |

На настоящей 1.7B-Base (сверка в 0.2–0.4): логиты talker ≤ 1.3e-4, predictor ≤ 2.6e-4, argmax 100% на 5 фразах.

Как устроена проверка: `vox verify` прогоняет эталонные коды «с учителем» и сравнивает логиты каждого шага и аудио
декодера, а затем делает свободную жадную генерацию через C ABI. Для q8 используйте `--quant 1`: средняя KL
(допуск 0.02), доля шагов с эталонным argmax в top-5 (≥ 97%) и аудио декодера.

## Полный прогон и сравнение

    python tools/run_full_check.py --model-dir ./Qwen3-TTS-12Hz-1.7B-Base --ref-audio ref.wav \
        --ref-text "Текст референса" --lang russian --dtypes bf16,q8 \
        --devices "cpu;vulkan:1,strict;vulkan:1,codec=cpu,strict" --threads 8 \
        --external "qwentts.cpp=./qwentts-cli --model m.gguf --out {out} < {text_file}" \
        --external-time 'qwentts.cpp=\[Perf\] Total:\s*([\d.]+)\s*ms' \
        --wer --asr-cmd 'whisper-cli -m ggml-small.bin -l {lang} -nt -np -f {wav}' --out work/

Внешняя команда выполняется через `sh -c`, поэтому работают `< {text_file}` и пайпы. `--external-time` берёт время
синтеза из вывода самого движка (группа 1 — миллисекунды). Без него RTF внешнего движка включает загрузку модели,
а у vox RTF всегда считается без неё. WER и CER считаются для каждой пары «формат × устройство»; числа перед
сравнением приводятся к словам («5» → «пять»). ASR — faster-whisper, whisper или любая команда из `--asr-cmd`.

Что делает прогон:
1. конвертирует веса;
2. создаёт голос нативно и сравнивает его с голосом из `qwen_tts`;
3. снимает эталонные дампы PyTorch;
4. проверяет каждую пару «формат × устройство»;
5. меряет скорость: холодный и тёплый запуск, время до первого чанка, пик памяти, профиль по стадиям, GPU-профиль,
   размещение весов в VRAM;
6. запускает внешний движок на том же тексте;
7. с `--wer` оценивает разборчивость через whisper.

Итог — `work/report.md`.

Профили:
- `VOX_PROFILE=1`: время по стадиям на CPU (prefill, talker, predictor, прогрев и декодер, мс на кадр).
- `VOX_PROFILE=gpu`: метки времени GPU после каждого шейдера, сумма по «стадия/шейдер» (`talker/matmul.q8`,
  `codec.block3/convgemm.f32`, …). Это реальное время работы видеокарты, а не оценка.

## Форматы

- `.tmod`: заголовок `VOXM`, JSON-манифест (конфигурации стадий, точность, токенизатор, параметры генерации),
  блобы с выравниванием 4096 под mmap. Веса декодера кодека и энкодеров голоса по умолчанию f32 (`--codec-dtype`).
- `.tvoice`: safetensors, внутри `speaker_emb`, `ref_codes`, `ref_text_tokens`, исходный WAV и параметры по умолчанию
  (seed, температура, …), привязка к весам по sha256. Голос режима `speaker` (готовый диктор / описание) хранит имя
  диктора и инструкцию; он привязан к варианту модели, потому что номера дикторов у моделей разные.
- Q4: 4 бита на вес, блоки по 32 (масштаб f32, значение `(u − 8)·масштаб`, масштаб блока подбирается по минимуму ошибки).
  Три схемы: `--quant q4` — 4 бита у внимания и MLP основного talker, predictor/головы/text_projection в q8;
  `q4mlp` — 4 бита только у MLP talker; `q4full` — всё в 4 бита (на реальной модели 0.8 дала KL predictor 0.2–0.4, вне
  допуска). Замер 0.8.1 на RTX 3050: GPU RTF 0.45 у q4 против 0.48 у q8, VRAM на треть меньше, но ни один q4 не проходит штатный допуск KL ≤ 0.02 / top-5 ≥ 97% (у q8 проходит), WER на пяти фразах 0.12–0.18 против 0.07; по прослушиванию клонов разницы не заметно. По умолчанию — q8. Качество смотреть так: `run_full_check.py --dtypes q8,q4,q4mlp,q4full`.

### F5-TTS (с 0.9)

    python -m vox.build_f5 --ckpt F5TTS_v1_Base/model_1250000.safetensors --vocab F5TTS_v1_Base/vocab.txt \
        --vocos vocos-mel-24khz/ --dtype bf16 --quant q8 -o f5.tmod
    vox voice create ref.wav --model f5.tmod --text "Текст, который сказан в ref.wav" -o me.tvoice
    vox say "Текст для озвучки" --model f5.tmod --voice me.tvoice -o out.wav [--nfe 32 --cfg 2 --sway -1 --speed 1]

Модель не авторегрессионная, поэтому она идёт через «прямой» режим `Backend` (`direct()`, `prepare_voice`, `direct_text`,
`direct_step`): текст режется на куски так же, как в официальном `infer_process`, каждый кусок озвучивается целиком
(NFE шагов Эйлера с CFG и sway-сэмплингом, затем Vocos), куски склеиваются кроссфейдом 0.15 с. Сессия отдаёт звук по мере
готовности кусков. Параметры называются как в официальном коде (`nfe_step`, `cfg_strength`, `sway_sampling_coef`,
`speed`, `cross_fade_duration`, `target_rms`, `use_epss`, `max_chars`) и задаются через `extra_json` / `--extra` /
`extra={...}` в Python; `--nfe --cfg --sway --speed` — короткие флаги CLI. Шум для каждого куска берётся из Philox по
(seed, номер куска), так что результат зависит только от seed. Голос хранит mel образца (`feat.mel`) и его громкость; голос
другой модели пересобирается из записи, лежащей внутри `.tvoice`.

Проверено на крошечной модели со случайными весами против настоящих `f5_tts` и `vocos` (`tests/test_f5_tiny.py`): токенизация,
mel образца, текстовый энкодер, DiT (обе ветки CFG), Vocos и весь цикл CFM совпадают до ~1e-6, сквозной звук четырёх
сценариев (один кусок, нарезка с кроссфейдом, формула длины куска, `speed`) — до 3e-9 при том же шуме. **Не проверено:**
качество на настоящих весах (в разработке не было доступа к Hugging Face) и скорость на видеокарте. Известные отличия от
официального пайплайна: китайский текст переводится в пиньинь по иероглифам без словаря слов (нет контекстной омофонии и
сандхи тонов, например «你好» даёт `ni3 hao3`, а не `ni2 hao3`); нет очистки образца от тишины (pydub) — берутся первые
12 с; `fix_duration` не поддержан. Внимание DiT считается на CPU, линейные слои — на CPU или GPU (Vulkan); вокодер на CPU.

### Браузер: vox.js (WebGPU, с 0.10)

Один скрипт без зависимостей и сборки (`docs/vox.js`, около 75 КБ; тот же код как ES-модуль — `docs/vox.mjs`). Сейчас умеет семейство
F5-TTS: веса и все активации живут в видеопамяти, на CPU остаются токены, mel образца, вектор времени и iSTFT.

```html
<script src="vox.js"></script>
<script type="module">
  const model = await Vox.load("f5.bf16.tmod", { onProgress: (p) => console.log(p) });   // URL, File, Blob, ArrayBuffer
  const voice = await model.cloneVoice(refFile, "Текст, который сказан в записи");        // File, Blob, URL, Float32Array + sampleRate
  const { pcm, sampleRate } = await model.synthesize("Привет, мир!", { voice, nfe_step: 16, seed: 1 });
  Vox.play(pcm, sampleRate);                                                                // или Vox.wav(pcm, sampleRate) → Blob
  for await (const piece of model.stream("Длинный текст.", { voice })) { /* куски звука по мере готовности */ }
</script>
```

Параметры называются как в официальном F5-TTS (`nfe_step`, `cfg_strength`, `sway_sampling_coef`, `speed`, `cross_fade_duration`,
`target_rms`, `use_epss`, `max_chars`), отмена — `signal` (AbortSignal), `voice.export()` / `Vox.Voice.from(json)` — сохранение голоса.
Тот же `seed` даёт тот же звук, что `vox say --seed` в ядре: генератор шума Philox портирован в JS побитово.
Шейдеры WGSL лежат в `webgpu/wgsl/` (GEMM с разворачиванием f32/f16/bf16/q8/q4 на лету, LayerNorm, свёртки, RoPE, внимание с онлайн-softmax,
GRN), их собирает `tools/gen_wgsl.py`; библиотеку — `python tools/build_web.py` (части в `webgpu/web/src/`).
Нужен браузер с WebGPU (Chrome и Edge 113+ (Windows, macOS; на Linux WebGPU пока включается флагами), Firefox 141+ (Windows), Safari 26; сводка — [web.dev](https://web.dev/blog/webgpu-supported-major-browsers)), страница по HTTPS или с localhost. Для браузера берите bf16 или q8:
F5 в f32 весит больше 1 ГБ.

`tests/test_web.py` запускает vox.js в Chromium (WebGPU через программный SwiftShader) на крошечной модели и сверяет: токенизатор и mel образца с
ядром, сквозной звук четырёх сценариев с настоящим `infer_batch_process` (максимум |diff| 3.7e-9 при амплитуде 5e-3), собственный шум JS с шумом ядра
(тот же seed — тот же звук), bf16 и q8 с ядром, прикладной слой (экспорт голоса, поток, отмена, ошибки). **Не проверено:** скорость и поведение на
настоящих видеокартах и драйверах (SwiftShader считает на процессоре), настоящие веса F5. Известные ограничения: вычисления в f32 (без f16),
свёртки позиционного кодирования наивные, нет WASM-запасного пути без WebGPU, вся модель читается в память страницы целиком.
F5 тяжёлая: порядок 2 ТФЛОП на шаг и фразу в 15 секунд (обе ветки CFG), поэтому на слабых видеокартах ставьте `nfe_step` 7–16 (с EPSS).

### Сайт и демо

`site/` — исходники страницы проекта (презентация, как скачать, примеры по языкам, живое демо WebGPU); `python tools/build_site.py` собирает её в
`docs/`. Для GitHub Pages: Settings → Pages → Deploy from a branch → `main` / `/docs`. Примеры кода на странице подставляются прямо из `examples/`,
то есть это те же файлы, которые проверяют тесты. Демо принимает свой `.tmod` (файл или ссылка с CORS), запись или микрофон, а кнопка «Запустить тест»
прогоняет тестовую модель на 2 МБ со случайными весами (звук — шум, это проверка пути).

### Из других языков

Все привязки идут через C ABI. `examples/`: C (`minimal.c`), C++ (`minimal.cpp`), Rust (`rust/`, без зависимостей), Go (`go/`, cgo),
Node.js (`node/`, koffi), Python (`python/ctypes_example.py`, без установки пакета). `tests/test_examples.py` собирает и запускает каждый на
тестовой модели и требует, чтобы звук совпал с `vox say --seed 42` до бита (недостающие инструменты пропускаются).

### Модели: Base, CustomVoice, VoiceDesign

    vox voice speaker --model custom.tmod --speaker aiden --instruct "Speak slowly" -o aiden.tvoice
    vox say "Hello" --model design.tmod --instruct "A deep calm male voice" -o out.wav

`tts_model_type` читается из чекпойнта. Диктор — строка `codec_embedding`; «диалектные» дикторы подменяют язык
chinese/auto своим диалектом; инструкция идёт префиксом `<|im_start|>user\n…<|im_end|>\n`. Сверено с официальными
`generate_custom_voice` и `generate_voice_design` (логиты ≤ 2e-6). **Режим текста — потоковый** (как у клонирования,
`non_streaming_mode=False` в официальном коде); у официального CustomVoice по умолчанию `non_streaming_mode=True`,
там траектория генерации другая. 0.6B CustomVoice инструкций не принимает (как и в официальном коде).

## Устройство кода

| путь | что |
|---|---|
| `include/vox.h` | C ABI: engine / voice / session, pull-стриминг, отмена |
| `src/engine.cpp` | реализация ABI, сэмплинг, чанки, конвейер «GPU генерирует, CPU декодирует» |
| `src/backend_qwen3.cpp` | Qwen3-TTS: префикс ICL / x-vector / диктор / инструкция, talker, predictor, прогрев и режимы декодера |
| `src/f5.cpp`, `src/f5_ops.cpp`, `src/backend_f5.cpp` | F5-TTS: токенизатор, DiT + CFM, Vocos, mel/iSTFT/свёртки, нарезка текста и кроссфейд |
| `webgpu/wgsl/*.wgsl`, `webgpu/web/src/` | шейдеры WGSL и исходники `vox.js` (браузерный WebGPU-бэкенд F5-TTS); сборка — `tools/build_web.py` |
| `site/`, `docs/` | страница проекта и её сборка для GitHub Pages (`tools/build_site.py`) |
| `examples/` | клиенты C ABI на C, C++, Rust, Go, Node.js, Python ctypes |
| `src/resources.cpp` | менеджер ресурсов: конфигурация, система, планировщик, бюджет памяти сессий |
| `src/family.cpp` | реестр семейств моделей (C++ и Python), `vox plan` / `vox inspect` |
| `src/qwen3.cpp` | загрузка весов, потоковый декодер кодека на CPU |
| `src/qwen3_voice.cpp` | создание голоса: мел, ECAPA-TDNN, энкодер Mimi, RVQ, ресемплер |
| `src/kernels.cpp` | пулы потоков, matmul (f32/f16/bf16, q8 и q4 int8-dot AVX2), трансформер (в т.ч. гибрид GPU/CPU) |
| `src/gpu_vulkan.cpp`, `src/gpu_codec.cpp` | Vulkan: matmul, трансформер, декодер кодека, профайлер |
| `src/shaders/*.comp` | GLSL-шейдеры (SPIR-V в `src/vk_shaders.inc`) |
| `python/vox` | Python API, конвертеры `vox.build` (Qwen3) и `vox.build_f5` (F5), сервер `vox.serve` |
| `tools/` | эталоны (`dump_reference.py`, `export_voice.py`), полный прогон, крошечные модели |
| `tests/` | юнит-тесты, токенизатор, numpy-эталон, официальный эталон, API и сервер |

## Тесты

    tests/run_all.sh                                    # C++, токенизатор, F5-TTS против f5_tts (если стоят torch и f5-tts), примеры на 6 языках, vox.js в Chromium (если есть playwright), крошечная модель (в т.ч. гибрид и Vulkan), Python-библиотека,
                                                        # Python-семейства, сервер, официальный эталон (все варианты моделей), Q4
    VOX_QWEN2_GGUF=qwen2.gguf VOX_TEST_DEVICES="cpu;vulkan:0" python tests/test_official.py build/vox

Словарь для тестов: `curl -L -o qwen2.gguf https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/ggml-vocab-qwen2.gguf`.
Без него тесты токенизатора и официального эталона пропускаются с подсказкой. CI (`.github/workflows/ci.yml`)
гоняет всё это на CPU и на программном Vulkan (lavapipe).

## Ограничения

- GPU в нативных сборках: только Vulkan. CUDA и Metal не реализованы; нативный WebGPU (Dawn / wgpu-native) ещё не сделан: WebGPU есть только в браузере (`vox.js`) и только для F5-TTS. Шейдеры универсальные, без тензорных ядер и
  cooperative matrix. Один Vulkan-девайс на процесс.
- Скорость 0.8 на видеокарте не измерена (разработка шла без неё): Q4, гибридный talker и режимы ресурсов проверены на
  корректность (lavapipe + эталон), но не на скорость. Системные пробы RAM/ядер для macOS и Windows написаны, но не
  запускались.
- Основной путь рассчитан на авторегрессионные кодек-модели (talker → кодбуки → потоковый декодер). Модели «текст →
  звук кусками» (не-AR) подключаются через «прямой» режим `Backend`; так сделан F5-TTS. XTTS, CosyVoice и Chatterbox ещё не
  подключены.
- F5-TTS: внимание DiT и вокодер считаются на CPU; на длинных фразах без GPU это медленно (веса ~340M параметров × 2 ветки
  CFG × NFE шагов). Активации на GPU между слоями не держатся — каждый matmul гонит данные туда и обратно.
- Python-бэкенд считает на CPU в интерпретаторе: он нужен для прототипов и тестов, а не для скорости.
- Режим `non_streaming_mode=True` официального кода (весь текст в префилле) не реализован.
- Сэмплинг на GPU считает вероятности во float (на CPU — в double). Тот же seed на том же устройстве даёт тот же звук.
  Между CPU и GPU коды могут изредка расходиться, но это и так происходит из-за разницы логитов в последних битах.
- Ресемплинг входа для голоса — sinc/Kaiser. Побитовое совпадение с `qwen_tts` (soxr) гарантировано только для WAV
  24 кГц. На других частотах эмбеддинг чуть отличается (косинус ≈ 0.998), коды обычно совпадают.
- Нормализация NFC выполняется в Python-обёртке. Из CLI и C ABI передавайте текст уже в NFC.

## Лицензия

Apache-2.0 (см. `LICENSE`). `third_party/json.hpp` — nlohmann/json, MIT. Веса Qwen3-TTS распространяются на условиях
своей модели и в репозиторий не входят.
