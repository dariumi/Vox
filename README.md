**English** | [Русский](README.ru.md)

# vox

A speech synthesis engine in C++17: C ABI (`include/vox.h`), the `vox` CLI, the `vox` Python **library** and an HTTP server
with streaming. No PyTorch, no ggml, no external libraries: weights are read via mmap from a single `.tmod` file, and a
voice is stored in `.tvoice`. Backends: CPU (x86-64 AVX2, aarch64), Vulkan (NVIDIA / AMD / Intel / Android) and WebGPU in the browser (`vox.js`, for F5-TTS only so far). Weight formats:
f32 / f16 / bf16 / q8 / q4. Built-in model families: **Qwen3-TTS-12Hz** (Base: voice cloning;
CustomVoice: ready-made speakers and style instructions; VoiceDesign: a voice from a description; 0.6B and 1.7B) and **F5-TTS**
(DiT + flow matching + Vocos, cloning from a sample and its text; not autoregressive). Other models
plug in as a "family" (C++ or Python); the runtime stays the same. Queue: XTTS, CosyVoice, Chatterbox.

- **Resources under control**: what to compute on, how much RAM and VRAM to use, how many threads and at what priority,
  the operating mode (speed / balanced / memory / background), a limit on concurrent sessions. The placement plan can be inspected
  without loading anything (`vox plan`, `vox.plan()`).

- **The whole pipeline is native**: voice creation (mel spectrogram, ECAPA-TDNN, Mimi encoder with RVQ), talker, code predictor,
  streaming codec decoder. torch is needed only for reference checks.
- **Matches the official implementation** (`qwen-tts` 0.1.1): see "What is verified".
- **Deterministic**: sampling uses the counter-based RNG Philox. The result depends only on the seed and not on the number
  of threads or the chunk size.
- **Streaming**: text is fed in pieces, audio is pulled in chunks of 4 frames (320 ms). The decoder state
  is carried across chunks, so there are no seams.

> Status: correctness has been checked against the official code, including on a real 1.7B-Base. A 0.7 measurement
> on the user's machine (Ryzen 7 5700X, RTX 3050 6 GB, ~13 s of speech): Vulkan q8 RTF 0.47, bf16 0.62, CPU q8 1.82; qwentts.cpp:
> CUDA 0.428, CPU 2.07. **0.8 speed was not measured on hardware**: the new features (q4, hybrid talker, resource modes) were checked
> for correctness, and the speed figures will come from `tools/run_full_check.py` on your machine. Details and boundaries are in the CHANGELOG
> and in "Limitations".

## Build

    cmake -S . -B build -DVOX_NATIVE=ON && cmake --build build -j
    pip install .                     # Python module (nanobind, scikit-build-core, numpy)

Vulkan is enabled automatically if the headers are found (`libvulkan-dev`). SPIR-V shaders are stored in the repository
(`src/vk_shaders.inc`); glslang is needed only when they are changed (`python tools/gen_shaders.py`).
Termux: `pkg install clang cmake python python-numpy`, then the same steps. The aarch64 build is checked under qemu
(tests and comparison with the reference); it has not been run on a real phone.

## Quick start

    # 1. Hugging Face model → .tmod (q8 is the fastest on CPU; bf16 is unquantized)
    python -m vox build --model ./Qwen3-TTS-12Hz-1.7B-Base --dtype bf16 --quant q8 -o model.q8.tmod

    # 2. voice from a recording, natively, without torch (24 kHz WAV is best, 5–15 s of clean speech)
    vox voice create ref.wav --model model.q8.tmod --text "Текст, который звучит в записи" \
        --name my_voice --lang russian --seed 42 -o voices/my_voice.tvoice

    # 3. synthesis
    vox say "Привет! Это проверка." --model model.q8.tmod --voice voices/my_voice.tvoice -o out.wav
    vox say --stdin --stream --model model.q8.tmod --voice voices/my_voice.tvoice | ffplay -f f32le -ar 24000 -i -

    # 4. server (WAV/PCM streaming, voice creation via POST /v1/voices)
    python -m vox serve --model model.q8.tmod --voices ./voices --port 8080 --device vulkan

Python:

```python
import vox
eng = vox.Engine("model.q8.tmod", device="vulkan:0")
voice = eng.clone_voice("ref.wav", "Текст референса", name="my_voice", seed=42)
voice.save("my_voice.tvoice")
audio = eng.synthesize("Привет!", voice)                 # numpy float32, 24 kHz
with eng.session(voice, seed=42) as s:                    # text in pieces → audio as it becomes ready
    s.feed("Первая фраза. "); chunks = list(s.ready())
    s.end(); chunks += list(s)
```

### Python as a library (no CLI)

```python
import vox

vox.devices()                                   # CPU, RAM (respecting the container limit), GPUs and their free memory
vox.plan("model.q8.tmod", vox.Resources(device="gpu:1", vram_limit="3GiB"))   # what goes where, weights are not read
vox.inspect("model.q8.tmod")                    # family, variant, stages, capabilities

res = vox.Resources(device="gpu:1", mode="balanced", ram_limit="6GiB", vram_limit="3GiB", threads=6,
                    cpu_affinity="0-5", priority="low", place={"codec": "cpu"}, max_sessions=2)
with vox.Engine("model.q8.tmod", resources=res) as eng:
    eng.plan, eng.capabilities, eng.speakers, eng.languages, eng.stats()
    voice = eng.clone_voice("ref.wav", "Текст референса")            # Base
    voice = eng.speaker_voice("Aiden", instruct="Speak slowly")      # CustomVoice (1.7B: with an instruction)
    voice = eng.design_voice("A deep calm male voice")               # VoiceDesign
    wavs = eng.synthesize_many(["Первый.", "Второй."], voice, seed=1)   # parallel sessions
    eng.benchmark("Текст замера.", voice, runs=3)                    # RTF, first chunk, statistics, no CLI needed
```

Errors are `vox.VoxError`; running out of memory or hitting the session limit is `vox.OutOfMemory`, a missing device is `vox.DeviceError`.
You can write a model entirely in Python and register it (`vox.register_family`): see
[docs/python-backends.md](docs/python-backends.md).

To check a reference before creating a voice: `python tools/check_reference.py ref.wav --ref-text "..."`. The script looks at
the speaking rate and the pause at the end, and with whisper also at any mismatch between the text and the speech. An extra word at the start of the synthesis almost always
means that the reference text does not match the recording.

## Resources: device, memory, mode

    vox devices [--json]                          # CPU, RAM, GPUs with indices and free video memory
    vox plan --model m.tmod --mode memory --ram-limit 3GiB --vram-limit 2GiB     # plan without loading weights
    vox say "…" --model m.tmod --voice v.tvoice --device gpu:1 --mode balanced --threads 6 --affinity 0-5 \
        --priority low --ram-limit 6GiB --vram-limit 3GiB --place codec=cpu --gpu-layers talker=14 --max-sessions 2
    --config resources.json                       # the same as a JSON file; command-line flags override it

- **The planner** (`src/resources.cpp`) takes the model description (talker / predictor / codec stages: weight, traffic per frame),
  the system (cores, RAM accounting for cgroup, free memory of each video card) and the requested settings, and builds a plan: device,
  placement of each stage, threads, RAM and VRAM estimate. Stages with the greatest benefit per byte go to the GPU first.
  If the talker does not fit entirely, it is **split by layers**: the first k layers and their KV cache on the GPU, the rest on the CPU.
  Free video memory is taken from `VK_EXT_memory_budget` (an estimate if the extension is absent). Software Vulkan renderers
  (llvmpipe) are not chosen by `auto`.
- **Modes**: `speed` uses all cores with busy-waiting threads; `balanced` (default) leaves a core to the OS;
  `memory` uses half as many threads and no busy-waiting; `background` uses a quarter of the cores at low priority and does not go to the GPU.
- **Limits are hard**: if the plan exceeds `ram_limit` / `vram_limit`, the model is refused (`OutOfMemory` with the plan in
  the error text), unless `enforce=false` is set. A session that does not fit in the remainder of the limit (KV cache and decoder
  state) is not created; `max_sessions` caps their number. After loading to the GPU, the mmap pages holding the weights
  are returned to the system (`release_host_weights`, `--no-release` turns it off).
- Each engine has its own CPU threads (a pool pinned to cores, `nice`): two engines in one process with different thread counts
  give the same result (verified by a test).
- Limitation: one Vulkan device per process (several video cards in one process are not shared).

### Devices

    vox devices                                   # Vulkan devices with indices
    --device cpu | auto | gpu[:N] | vulkan | vulkan:N
    --device vulkan:1,strict                      # error if any matrix does not fit in VRAM
    --device vulkan:1,codec=cpu                   # codec decoder on the CPU (in parallel with the GPU), everything else on the GPU

- Vulkan numbering does not match nvidia-smi and CUDA; identify devices by the name from `vox devices`.
- At startup vox writes to stderr how many matrices went to VRAM, how many stayed on the CPU, and why (`VOX_VERBOSE=1`
  prints this always). There is no silent fallback; with `strict` (or `VOX_GPU_STRICT=1`) a lack of video memory becomes an error.
- **On the GPU**: the talker and predictor entirely (norms, RoPE, attention with GQA, MLP, input projection and head); layer weights
  are laid out contiguously (q|k|v and gate|up). The talker takes one dispatch per step. **The whole predictor loop for a frame (15 passes) is also
  a single dispatch**: sampling happens in the shader (the same algorithm and Philox as on the CPU), and the embedding of the chosen code is taken
  on the GPU. In total, 2 host synchronizations per frame (it was 16). Prefill with m > 8 tokens uses a tiled GEMM rather than GEMV.
  GPU sampling can be disabled with `VOX_GPU_SAMPLING=0`. **The codec decoder is also entirely on the GPU**:
  convolutional implicit-GEMM, Snake, ConvNeXt, transposed convolutions with overlap-add, a sliding-window transformer
  on a ring KV cache. One command buffer per chunk; only PCM goes out.
- `codec=cpu` keeps the decoder on the CPU. Then chunk k is decoded in a separate thread while the GPU computes chunk k+1
  (turn off with `VOX_NO_OVERLAP=1`). Which mode is faster on your card is shown by `bench` (below).

## Decoder modes

- `--decoder stream` (default): continuous streaming decoding, warmed up with the last 50 frames of the reference
  (`--context N|all`). Minimal latency, no seams even on long phrases.
- `--decoder reference`: bit-for-bit the same scheme as in `qwen_tts` (`chunked_decode`: pieces of 300 frames, 25 frames
  of context, warm-up from the start of the piece). Needed for verification and A/B comparisons with PyTorch and qwentts.cpp.

The difference between the modes is not an error margin. From frame 300 onward, the official code decodes "reference + speech" anew,
with a short context, so the sound after the boundary is slightly different. This is exactly what produced the 5.756e-2 error in the 0.4 report.

## What is verified

`tests/test_official.py` uses **the official `qwen_tts` classes** to build a tiny model with random weights
(`tools/make_tiny_official.py`: the same modules, tensor names and frame steps as the 1.7B, only narrow) and compares vox
with its `generate_voice_clone` and `create_voice_clone_prompt`:

| what | result (CPU, Vulkan with the decoder on the GPU and on the CPU) |
|---|---|
| speaker embedding (native) | rel. error ≤ 4e-7 |
| reference codes (native, Mimi + RVQ) | 100% match, including a 25 s reference (> 300 frames) |
| talker / predictor logits, f32 | ≤ 1e-6 / ≤ 1e-5, argmax 100% |
| decoder audio, f32 | ≤ 3e-4 (the official f32 itself differs from its own f64 by ~1e-4) |
| decoder chunk boundary inside the speech and inside the reference | matches (`reference` mode) |
| f16 / bf16 / q8 | format tolerances; q8 by KL and top-5 instead of argmax |
| aarch64 (qemu) | same results |

On a real 1.7B-Base (comparison in 0.2–0.4): talker logits ≤ 1.3e-4, predictor ≤ 2.6e-4, argmax 100% on 5 phrases.

How the check works: `vox verify` runs the reference codes with teacher forcing and compares the logits of every step and the decoder
audio, and then does free greedy generation through the C ABI. For q8 use `--quant 1`: mean KL
(tolerance 0.02), the share of steps where the reference argmax is in the top-5 (≥ 97%), and the decoder audio.

## Full run and comparison

    python tools/run_full_check.py --model-dir ./Qwen3-TTS-12Hz-1.7B-Base --ref-audio ref.wav \
        --ref-text "Текст референса" --lang russian --dtypes bf16,q8 \
        --devices "cpu;vulkan:1,strict;vulkan:1,codec=cpu,strict" --threads 8 \
        --external "qwentts.cpp=./qwentts-cli --model m.gguf --out {out} < {text_file}" \
        --external-time 'qwentts.cpp=\[Perf\] Total:\s*([\d.]+)\s*ms' \
        --wer --asr-cmd 'whisper-cli -m ggml-small.bin -l {lang} -nt -np -f {wav}' --out work/

The external command is run through `sh -c`, so `< {text_file}` and pipes work. `--external-time` takes the synthesis time
from the output of the engine itself (group 1 is milliseconds). Without it, the RTF of the external engine includes model loading,
while vox's RTF is always computed without it. WER and CER are computed for each "format × device" pair; numbers are converted
to words before comparison ("5" → "пять"). ASR is faster-whisper, whisper, or any command given in `--asr-cmd`.

What the run does:
1. converts the weights;
2. creates a voice natively and compares it with the voice from `qwen_tts`;
3. takes the PyTorch reference dumps;
4. checks each "format × device" pair;
5. measures speed: cold and warm start, time to first chunk, peak memory, per-stage profile, GPU profile,
   placement of weights in VRAM;
6. runs the external engine on the same text;
7. with `--wer`, estimates intelligibility via whisper.

The result is `work/report.md`.

Profiles:
- `VOX_PROFILE=1`: per-stage time on the CPU (prefill, talker, predictor, warm-up and decoder, ms per frame).
- `VOX_PROFILE=gpu`: GPU timestamps after each shader, summed by "stage/shader" (`talker/matmul.q8`,
  `codec.block3/convgemm.f32`, …). This is the real time the video card spends working, not an estimate.

## Formats

- `.tmod`: `VOXM` header, JSON manifest (stage configurations, precision, tokenizer, generation parameters),
  blobs aligned to 4096 for mmap. The weights of the codec decoder and the voice encoders are f32 by default (`--codec-dtype`).
- `.tvoice`: safetensors, containing `speaker_emb`, `ref_codes`, `ref_text_tokens`, the source WAV and default parameters
  (seed, temperature, …), bound to the weights by sha256. A voice in `speaker` mode (ready-made speaker / description) stores the speaker
  name and the instruction; it is tied to the model variant, because speaker numbers differ between models.
- Q4: 4 bits per weight, blocks of 32 (f32 scale, value `(u − 8)·scale`, the block scale is chosen to minimize the error).
  Three schemes: `--quant q4` uses 4 bits for the attention and MLP of the main talker, with predictor/heads/text_projection in q8;
  `q4mlp` uses 4 bits only for the talker MLP; `q4full` puts everything in 4 bits (on the real model 0.8 gave a predictor KL of 0.2–0.4, outside
  tolerance). Measurement of 0.8.1 on an RTX 3050: GPU RTF 0.45 for q4 against 0.48 for q8, a third less VRAM, but no q4 passes the standard tolerance KL ≤ 0.02 / top-5 ≥ 97% (q8 does), WER on five phrases 0.12–0.18 against 0.07; no difference is noticeable when listening to the clones. The default is q8. To assess quality: `run_full_check.py --dtypes q8,q4,q4mlp,q4full`.

### F5-TTS (since 0.9)

    python -m vox.build_f5 --ckpt F5TTS_v1_Base/model_1250000.safetensors --vocab F5TTS_v1_Base/vocab.txt \
        --vocos vocos-mel-24khz/ --dtype bf16 --quant q8 -o f5.tmod
    vox voice create ref.wav --model f5.tmod --text "Текст, который сказан в ref.wav" -o me.tvoice
    vox say "Текст для озвучки" --model f5.tmod --voice me.tvoice -o out.wav [--nfe 32 --cfg 2 --sway -1 --speed 1]

**F5TTS_Base (v0) and its fine-tunes** (for example `hotstone228/F5-TTS-Russian`) differ from `F5TTS_v1_Base` in the config, not in the weights:
`text_mask_padding=False` and RoPE only on the first head (`pe_attn_head=1`). The converter guesses it from the path (`F5TTS_Base`/`F5TTS_Small` → v0, otherwise v1);
set it explicitly with `--model-cfg v0|v1` (or `--text-mask-padding`, `--pe-attn-head`). In the browser: `Vox.load(url, { modelCfg: 'v0' })`; the result is in `model.info.f5`.
A wrong config gives noise instead of speech. Models converted before 0.12.1 are read as v1 — reconvert v0 models.
Since 0.12.2 the converter also reads `vocab.txt` with CRLF line endings correctly (before, every token got a trailing `\r` and the text was lost) and takes the
ISTFT padding of the vocoder from its `config.yaml` (`center` for `charactr/vocos-mel-24khz`; before, the audio was shifted by 128 samples). Reconvert models made with 0.12.1 or older.

The model is not autoregressive, so it goes through the "direct" mode of `Backend` (`direct()`, `prepare_voice`, `direct_text`,
`direct_step`): the text is split into pieces the same way as in the official `infer_process`, each piece is synthesized as a whole
(NFE Euler steps with CFG and sway sampling, then Vocos), and the pieces are joined with a 0.15 s crossfade. A session hands out audio as
the pieces become ready. The parameters are named as in the official code (`nfe_step`, `cfg_strength`, `sway_sampling_coef`,
`speed`, `cross_fade_duration`, `target_rms`, `use_epss`, `max_chars`) and are set via `extra_json` / `--extra` /
`extra={...}` in Python; `--nfe --cfg --sway --speed` are short CLI flags. The noise for each piece is taken from Philox by
(seed, piece number), so the result depends only on the seed. A voice stores the mel of the sample (`feat.mel`) and its loudness; a voice
of another model is rebuilt from the recording stored inside the `.tvoice`.

Verified on a tiny model with random weights against the real `f5_tts` and `vocos` (`tests/test_f5_tiny.py`): tokenization,
sample mel, text encoder, DiT (both CFG branches), Vocos and the whole CFM loop match to ~1e-6; end-to-end audio for four
scenarios (one piece, splitting with crossfade, the piece length formula, `speed`) matches to 3e-9 with the same noise. **Not verified:**
quality on real weights (there was no Hugging Face access during development) and speed on a video card. Known differences from the
official pipeline: Chinese text is converted to pinyin character by character without a word dictionary (no contextual homophony and
tone sandhi, e.g. "你好" gives `ni3 hao3`, not `ni2 hao3`); there is no silence trimming of the sample (pydub), the first
12 s are taken; `fix_duration` is not supported. DiT attention is computed on the CPU, linear layers on the CPU or GPU (Vulkan); the vocoder is on the CPU.

### Duplex streaming (since 0.11)

Text can be fed on the fly and audio pulled as it becomes ready, and also the other way round: type text and get one single recording.
Models do not all behave the same, so the engine takes over the "plumbing": it knows how the model accepts text and decides what to give it and when.

| Model | how it reads text | what the engine does |
|---|---|---|
| Qwen3-TTS | word by word (`text_in: token`): speaking can start mid-sentence | hands over words as they arrive, holds back an unfinished word; an utterance is closed by `utterance_gap_ms` |
| F5-TTS | whole pieces (`text_in: chunk`): it needs the duration of the entire piece | accumulates text up to the end of a sentence (or a typing pause `idle_ms`), synthesizes the pieces in turn; with `auto` the result matches synthesizing the whole text at once |

Commit policy (`commit`): `auto` is whatever the model can do; `sentence` is complete sentences only; `manual` means nothing goes out until `flush()` / `end()`
("I type, then I get one recording"). `idle_ms` is an auto-commit after a pause in typing, `utterance_gap_ms` closes the utterance after a pause.
Typing mode: `set_text(whole_text_from_the_field)`: additions and edits to the not-yet-spoken tail are applied, what has already been spoken is not re-spoken
(`on_conflict`: `keep` | `raise` | `respeak`). Model capabilities: `engine.capabilities["stream"]` (`text_in`, `audio_out`, `exact`).

    # C ABI: {"stream": {"commit": "auto", "idle_ms": 600}} in the session's extra_json; vox_session_feed_text / _flush / _set_pending / _end_text / _pull
    with engine.duplex(voice, commit="auto", idle_ms=600) as d:       # Python
        d.write("Привет, "); d.write("как дела?"); d.end()           # from any thread
        for chunk in d.audio(): play(chunk)                           # aaudio() is for asyncio, final() gives one recording
    printf 'Привет. Как дела?' | vox live --model m.tmod --voice v.tvoice -o out.wav --commit sentence   # CLI, text from stdin
    # HTTP: WebSocket /v1/duplex — {"type":"start",…} {"type":"text"|"set_text"|"flush"|"end"|"cancel"}, audio arrives as binary PCM frames

The same interface in the browser: `model.duplex({ voice, commit, idle_ms })` → `write` / `setText` / `flush` / `end` / `cancel`, `for await (… of d.audio())`
(see below). A Python example is `examples/python/duplex_example.py`. Tests are `tests/test_duplex.py` (core, Python, CLI, WebSocket) and `tests/test_web_qwen.py` (browser).
**Not verified:** latencies on real models and video cards (on the test models the audio starts immediately, but it is noise), behavior with real speech from an LLM stream.
Limitations: for an AR model one utterance is one session, so with `utterance_gap_ms` prosody is not carried over between utterances;
F5 cannot synthesize a piece before it has seen its end.

### Browser: vox.js (WebGPU, since 0.10; Qwen3-TTS and loading by link since 0.11; CPU fallback and Qwen Base voice cloning since 0.12)

A single script with no dependencies and no build step (`docs/vox.js`, about 230 KB; the same code as an ES module is `docs/vox.mjs`). It supports the F5-TTS and Qwen3-TTS families:
the weights and all activations live in video memory; tokens, embedding tables, sampling, the sample mel and the iSTFT stay on the CPU.

**Model by link.** `Vox.load` accepts a Hugging Face repository (`hf:org/name`, `org/name`, a page link, `tree` or `resolve`), a direct link, a `.tmod`,
a `File`/`Blob`/bytes, or a set of files `{files: {"config.json": File, …}}`. If it is not a ready `.tmod` but a checkpoint (safetensors; for F5 also `.pt`),
it is downloaded and converted right in the browser, with the same transformations as `python -m vox.build` / `vox.build_f5` (for Qwen3 verified byte for byte for every tensor).
After conversion F5 is put into the browser's Cache API. Qwen3-TTS is read tensor by tensor (HTTP Range using the safetensors header, six requests in parallel),
and each tensor goes straight to video memory, so the whole model is never assembled in page memory (only the text embedding table stays on the CPU).
Reading is done in large blocks (8 MiB by default) with six requests in parallel, rather than one request per tensor (Qwen3-TTS has more than 400 of them, and with the real HF every request is also a redirect to the CDN plus a CORS preflight);
after the first redirect the engine goes directly to the CDN address. Blocks are put into the Cache API (key: address, size and ETag of the file), so a second load of the same model downloads nothing.
Load statistics are in `model.info.net` (requests, MiB, redirects, blocks from the cache).
Options: `dtype` (`bf16` | `f16` | `f32`), `quant` (`q8`; `q4` only via Python), `token` (private repositories), `hfHost` (mirror), `onProgress`, `onStatus`,
`backend` (`auto` | `webgpu` | `cpu`), `load` (`balanced` | `gentle` | `full`), `cache: false` (do not store downloaded blocks), `blockMiB` (read block size, 8).
`Vox.convert(src)` returns the bytes of a `.tmod` to save to a file (memory for the whole model is required).

```html
<script src="vox.js"></script>
<script type="module">
  // F5-TTS: a ready .tmod, a checkpoint by link, or an HF repository
  const f5 = await Vox.load("hf:SWivid/F5-TTS", { dtype: "bf16", onProgress: (p) => console.log(p) });
  const voice = await f5.cloneVoice(refFile, "Текст, который сказан в записи");              // File, Blob, URL, Float32Array + sampleRate
  const { pcm, sampleRate } = await f5.synthesize("Привет, мир!", { voice, nfe_step: 16, seed: 1 });
  Vox.play(pcm, sampleRate);                                                                    // or Vox.wav(pcm, sampleRate) → Blob

  // Qwen3-TTS: speaker / voice description (CustomVoice, VoiceDesign) or .tvoice (Base)
  const q = await Vox.load("hf:Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", { dtype: "bf16" });
  const v = q.speakerVoice(q.speakers[0], "", { language: "english" });                        // q.voiceFromFile(bytesOfTvoice) is for Base
  for await (const piece of q.stream("Long text.", { voice: v, seed: 1 })) { /* pieces of audio as they become ready */ }

  // Qwen3-TTS Base: clone a voice straight from a recording (File, Blob, URL, Float32Array + sampleRate) and the recording's text
  // const base = await Vox.load("hf:Qwen/Qwen3-TTS-12Hz-0.6B-Base"); const cv = await base.cloneVoice(file, "Что сказано в записи"); download(cv.toTvoice());

  // duplex streaming
  const d = q.duplex({ voice: v, commit: "auto", idle_ms: 600 });
  input.oninput = () => d.setText(input.value);                                                 // or d.write(a token from an LLM)
  (async () => { for await (const piece of d.audio()) player.push(piece); })();
</script>
```

The F5 parameters are named as in the official code (`nfe_step`, `cfg_strength`, `sway_sampling_coef`, `speed`, `cross_fade_duration`,
`target_rms`, `use_epss`, `max_chars`); cancellation is via `signal` (AbortSignal); `voice.export()` / `Vox.Voice.from(json)` save a voice.
The same `seed` gives the same audio as `vox say --seed` in the core: the Philox noise generator is ported to JS bit for bit (for Qwen, sampling too).
The WGSL shaders are in `webgpu/wgsl/` (GEMM with on-the-fly unpacking of f32/f16/bf16/q8/q4, LayerNorm and RMSNorm, convolutions, RoPE, attention with online softmax and with a KV cache,
transposed convolutions of the streaming decoder, Snake, GRN) and are generated by `tools/gen_wgsl.py`; the library is built by `python tools/build_web.py` (parts in `webgpu/web/src/`).
**Without WebGPU** vox.js switches to the CPU by itself (`backend: "auto"`, forced with `backend: "cpu"`): the same shaders are implemented in JS (`CpuGpu`), run in slices of ~24 ms and yield the thread to the page, so the tab does not freeze. This is orders of magnitude slower than a video card and runs on a single core: it is suitable for the test model, F5 with a small `nfe_step`, and short Qwen3-TTS 0.6B phrases; `model.info.backend` tells what exactly is computing.
**Video card load.** Commands are accumulated and submitted in slices (~30 ms; the slice size is tuned from the measured time), with a pause between slices so that the GPU is not busy for more than ~70% of the time (`load: "balanced"`; `gentle` is ~40%, `full` is no pauses). This way the OS and the browser do not freeze and the driver watchdog timer does not fire; `model.gpu.setLoad(...)` changes the mode on the fly. A single very heavy dispatch cannot be split into slices, so on very weak integrated video cards `gentle` or `cpu` is better. Shaders are compiled asynchronously (`createComputePipelineAsync`) in parallel with downloading the weights.
It works best with WebGPU (Chrome and Edge 113+ (Windows, macOS; on Linux WebGPU still has to be enabled with flags), Firefox 141+ (Windows), Safari 26; summary: [web.dev](https://web.dev/blog/webgpu-supported-major-browsers)), with the page served over HTTPS or from localhost.
For the browser use bf16 or q8: F5 in f32 weighs more than 1 GB.

`tests/test_web.py` runs vox.js in Chromium (WebGPU via software SwiftShader) on a tiny F5 and checks: the tokenizer and sample mel against
the core, end-to-end audio of four scenarios against the real `infer_batch_process` (maximum |diff| 3.7e-9 at an amplitude of 5e-3), JS's own noise against the core's noise,
bf16 and q8 against the core, the application layer. `tests/test_web_convert.py` checks loading and conversion of F5 against a mock "Hugging Face" server.
`tests/test_web_qwen.py` covers Qwen3-TTS (tiny CustomVoice and VoiceDesign from the official classes): conversion in JS byte for byte = `python -m vox.build` (bf16 and f16+q8),
loading by HF link in Range pieces, audio = `vox say --seed` (also compared with an official PyTorch dump during development: ICL, x-vector, speaker with an instruction),
JS tokenizer = core, duplex streaming (auto, sentence, manual, pause), reading in blocks through a "CDN" redirect and reloading from the cache, Base voice cloning from a recording (speaker vector 1e-7, Mimi codes match, audio = `vox say --voice`).
`tests/test_web_cpu.py` covers vox.js without WebGPU: the CPU backend = core (f32, bf16, f16; divergence < 1e-4), q8 = WebGPU, the page does not freeze; for F5 the same is in `tests/test_web.py` (divergence 1.4e-9).
**Not verified:** speed and behavior on real video cards and drivers (SwiftShader computes on the CPU and is very slow), real Qwen3-TTS 0.6B/1.7B weights
and F5 in the browser (tiny models with random weights were checked), Safari and Firefox, loading from the real huggingface.co (the tests use a mock server).
Known limitations: computation is in f32 (no f16 and no cooperative matrix), the KV cache is in f32 (1.7B at length 3072 is about 0.7 GB), a single token goes through a full GEMM tile (there is no separate GEMV kernel),
Qwen has 16 video card round trips per frame (one per codebook), voice creation from a recording for Qwen Base (`model.cloneVoice(audio, refText)`: mel → ECAPA → Mimi, all on the CPU with the thread yielded, `voice.toTvoice()` saves the voice) was checked on a tiny model and its time on the real Mimi was not measured, q4 is not built in the browser, the `reference` decoder mode and the warm-up cache of ICL voices are not implemented, the CPU fallback runs in JS on a single thread (no WASM SIMD and no workers), and q8/bf16 of the text embedding table stays in the page's RAM.
F5 is heavy: on the order of 2 TFLOP per step for a 15-second phrase (both CFG branches), so on weak video cards set `nfe_step` to 7–16 (with EPSS).

### Site and demo

`site/` holds the sources of the project page (presentation, how to download, examples by language, a live WebGPU demo); `python tools/build_site.py` builds it into
`docs/` (English `index.html` and Russian `index.ru.html`; interface strings live in `site/i18n.en.js` and `site/i18n.ru.js`). For GitHub Pages: Settings → Pages → Deploy from a branch → `main` / `/docs`. The code examples on the page are inserted directly from `examples/`,
so these are the same files that the tests check. The demo accepts a Hugging Face repository (weights are downloaded and converted in the browser), your own `.tmod` (a file or a link with CORS), a recording or a microphone; for Qwen3-TTS there is a choice of speaker and a voice description; the "Live input" section shows duplex streaming, and the "Run test" button
runs a 2 MB test model with random weights (the audio is noise, this checks the path).

### From other languages

All bindings go through the C ABI. `examples/`: C (`minimal.c`), C++ (`minimal.cpp`), Rust (`rust/`, no dependencies), Go (`go/`, cgo),
Node.js (`node/`, koffi), Python (`python/ctypes_example.py`, no package installation needed). `tests/test_examples.py` builds and runs each one on a
test model and requires the audio to match `vox say --seed 42` bit for bit (missing tools are skipped).

### Models: Base, CustomVoice, VoiceDesign

    vox voice speaker --model custom.tmod --speaker aiden --instruct "Speak slowly" -o aiden.tvoice
    vox say "Hello" --model design.tmod --instruct "A deep calm male voice" -o out.wav

`tts_model_type` is read from the checkpoint. A speaker is a row of `codec_embedding`; "dialect" speakers replace the language
chinese/auto with their own dialect; the instruction goes in as the prefix `<|im_start|>user\n…<|im_end|>\n`. Verified against the official
`generate_custom_voice` and `generate_voice_design` (logits ≤ 2e-6). **The text mode is streaming** (as with cloning,
`non_streaming_mode=False` in the official code); the official CustomVoice defaults to `non_streaming_mode=True`,
where the generation trajectory is different. 0.6B CustomVoice does not accept instructions (as in the official code).

## Code layout

| path | what |
|---|---|
| `include/vox.h` | C ABI: engine / voice / session, pull streaming, cancellation |
| `src/engine.cpp` | ABI implementation, sampling, chunks, the "GPU generates, CPU decodes" pipeline |
| `src/backend_qwen3.cpp` | Qwen3-TTS: ICL / x-vector / speaker / instruction prefix, talker, predictor, warm-up and decoder modes |
| `src/f5.cpp`, `src/f5_ops.cpp`, `src/backend_f5.cpp` | F5-TTS: tokenizer, DiT + CFM, Vocos, mel/iSTFT/convolutions, text splitting and crossfade |
| `webgpu/wgsl/*.wgsl`, `webgpu/web/src/` | WGSL shaders and the sources of `vox.js` (the browser WebGPU backend for F5-TTS); build with `tools/build_web.py` |
| `site/`, `docs/` | the project page and its build for GitHub Pages (`tools/build_site.py`) |
| `examples/` | C ABI clients in C, C++, Rust, Go, Node.js, Python ctypes |
| `src/resources.cpp` | resource manager: configuration, system, planner, session memory budget |
| `src/family.cpp` | registry of model families (C++ and Python), `vox plan` / `vox inspect` |
| `src/qwen3.cpp` | weight loading, streaming codec decoder on the CPU |
| `src/qwen3_voice.cpp` | voice creation: mel, ECAPA-TDNN, Mimi encoder, RVQ, resampler |
| `src/kernels.cpp` | thread pools, matmul (f32/f16/bf16, q8 and q4 int8-dot AVX2), transformer (including GPU/CPU hybrid) |
| `src/gpu_vulkan.cpp`, `src/gpu_codec.cpp` | Vulkan: matmul, transformer, codec decoder, profiler |
| `src/shaders/*.comp` | GLSL shaders (SPIR-V in `src/vk_shaders.inc`) |
| `python/vox` | Python API, converters `vox.build` (Qwen3) and `vox.build_f5` (F5), server `vox.serve` |
| `tools/` | references (`dump_reference.py`, `export_voice.py`), the full run, tiny models |
| `tests/` | unit tests, tokenizer, numpy reference, official reference, API and server |

## Tests

    tests/run_all.sh                                    # C++, tokenizer, F5-TTS against f5_tts (if torch and f5-tts are installed), examples in 6 languages, vox.js in Chromium (if playwright is available), tiny model (including hybrid and Vulkan), Python library,
                                                        # Python families, server, official reference (all model variants), Q4
    VOX_QWEN2_GGUF=qwen2.gguf VOX_TEST_DEVICES="cpu;vulkan:0" python tests/test_official.py build/vox

Vocabulary for the tests: `curl -L -o qwen2.gguf https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/ggml-vocab-qwen2.gguf`.
Without it, the tokenizer and official-reference tests are skipped with a hint. CI (`.github/workflows/ci.yml`)
runs all of this on the CPU and on software Vulkan (lavapipe).

## Limitations

- GPU in native builds: Vulkan only. CUDA and Metal are not implemented; native WebGPU (Dawn / wgpu-native) is not done yet: WebGPU exists only in the browser (`vox.js`: F5-TTS and Qwen3-TTS). The shaders are generic, without tensor cores and
  cooperative matrix. One Vulkan device per process.
- 0.8 speed has not been measured on a video card (development went on without one): Q4, the hybrid talker and the resource modes were checked for
  correctness (lavapipe + reference), but not for speed. The system RAM/core probes for macOS and Windows are written but have
  not been run.
- The main path is designed for autoregressive codec models (talker → codebooks → streaming decoder). "Text →
  audio in pieces" (non-AR) models plug in through the "direct" mode of `Backend`; F5-TTS is built this way. XTTS, CosyVoice and Chatterbox are not
  plugged in yet.
- F5-TTS: DiT attention and the vocoder are computed on the CPU; on long phrases without a GPU this is slow (~340M parameters of weights × 2 CFG branches
  × NFE steps). Activations are not kept on the GPU between layers: each matmul sends data back and forth.
- The Python backend computes on the CPU in the interpreter: it is meant for prototypes and tests, not for speed.
- The official code's `non_streaming_mode=True` mode (the whole text in the prefill) is not implemented.
- GPU sampling computes probabilities in float (on the CPU, in double). The same seed on the same device gives the same audio.
  Between CPU and GPU the codes may occasionally diverge, but that happens anyway because of differences in the logits in the last bits.
- Input resampling for the voice is sinc/Kaiser. A bit-exact match with `qwen_tts` (soxr) is guaranteed only for 24 kHz
  WAV. At other rates the embedding differs slightly (cosine ≈ 0.998), and the codes usually match.
- NFC normalization is done in the Python wrapper. From the CLI and the C ABI, pass text that is already in NFC.

## License

Apache-2.0 (see `LICENSE`). `third_party/json.hpp` is nlohmann/json, MIT. The Qwen3-TTS weights are distributed under the terms of
their own model and are not part of the repository.
