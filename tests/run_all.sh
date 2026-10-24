#!/bin/sh
# Полный локальный прогон: C++ тесты, токенизатор, Qwen3 на крошечной модели (CPU и, если есть, Vulkan),
# Python API, сервер и сверка с официальной реализацией qwen_tts (если установлены torch и qwen-tts).
# Нужно: собранный build/, pip install ., pip install numpy tokenizers regex и словарь Qwen2:
#   curl -L -o qwen2.gguf https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/ggml-vocab-qwen2.gguf
#   VOX_QWEN2_GGUF=qwen2.gguf tests/run_all.sh
set -e
cd "$(dirname "$0")/.."
export VOX_QWEN2_GGUF="${VOX_QWEN2_GGUF:-qwen2.gguf}"
(cd build && ./vox_tests)
python3 tests/test_tokenizer.py build/vox "$VOX_QWEN2_GGUF"
VOX_TEST_DEVICE=cpu python3 tests/test_qwen3_tiny.py build/vox
DEVICES="cpu"
if ./build/vox devices 2>/dev/null | grep -q '^vulkan:0'; then
  VOX_TEST_DEVICE=vulkan:0,strict python3 tests/test_qwen3_tiny.py build/vox
  VOX_TEST_DEVICE=vulkan:0,layers=1,strict python3 tests/test_qwen3_tiny.py build/vox   # гибрид: 1 слой talker на GPU, остальные на CPU
  DEVICES="cpu;vulkan:0,strict;vulkan:0,codec=cpu,strict;vulkan:0,layers=1,strict"
fi
python3 tests/test_python.py
F5_DEVICES="cpu"
if ./build/vox devices 2>/dev/null | grep -q '^vulkan:0'; then F5_DEVICES="cpu;vulkan:0,strict"; fi
VOX_TEST_DEVICES="$F5_DEVICES" python3 tests/test_f5_tiny.py build/vox   # F5-TTS против f5_tts (нужны torch, f5-tts, vocos; иначе SKIP)
python3 tests/test_kokoro_tiny.py build/vox   # Kokoro против официального kokoro + misaki (нужны torch, kokoro, misaki; иначе SKIP)
python3 tests/test_cosyvoice_tiny.py build/vox   # CosyVoice2 против официального кода (нужен клон CosyVoice в VOX_COSYVOICE_SRC; иначе SKIP)
python3 tests/test_library.py
python3 tests/test_examples.py build/vox   # примеры на C, C++, Rust, Go, Node.js, Python ctypes: звук = звуку CLI до бита (нет инструментов — SKIP)
VOX_WEB_FULL="${VOX_WEB_FULL:-}" python3 tests/test_web.py build/vox   # vox.js в Chromium (WebGPU) против эталона f5_tts (нужны playwright + chromium, torch, f5-tts; иначе SKIP)
python3 tests/test_web_convert.py   # загрузка и конвертация F5 в браузере против мок-HF (нужны node, torch, safetensors; иначе SKIP)
python3 tests/test_web_qwen.py build/vox   # Qwen3-TTS в браузере: HF-ссылка, конвертация побайтно = python, WebGPU, duplex (иначе SKIP)
python3 tests/test_web_cpu.py build/vox   # vox.js без WebGPU: процессорный бэкенд = ядро (f32/bf16/f16), q8 = WebGPU, страница не замирает (иначе SKIP)
python3 tests/test_duplex.py build/vox   # двусторонний стриминг: ядро, Python, CLI vox live, WebSocket
python3 tests/test_profiles.py build/vox   # профили: несколько голосов и стилей, разметка, один профиль на разных моделях, совместимость с 1.0
python3 tests/test_web_studio.py   # vox web: API Vox Studio, секрет, CORS, TLS, страница Studio в Chromium (если есть playwright)
python3 tests/test_py_family.py
VOX_TEST_DEVICES="$DEVICES" python3 tests/test_api_serve.py
VOX_TEST_DEVICES="$DEVICES" python3 tests/test_q4.py build/vox
VOX_TEST_DEVICES="$DEVICES" python3 tests/test_official.py build/vox
echo "ALL OK"
