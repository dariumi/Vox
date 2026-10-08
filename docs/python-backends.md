# Модель на Python: `vox.register_family`

Рантайм vox (сессии, потоковая подача текста, детерминированный сэмплинг Philox, лимиты ресурсов, `.tvoice`,
сервер, C ABI) не знает, что за модель под ним. Модель — это **семейство** (`Family`) и **бэкенд** (`Backend`).
Свою модель можно написать на Python и подключить без правок C++:

```python
import numpy as np, vox

class MyBackend:
    def __init__(self, tmod: vox.Tmod): ...        # tmod.manifest (dict), tmod.blob("имя") → numpy из mmap
    def info(self):                                  # параметры аудио и кодов
        return dict(sample_rate=24000, samples_per_frame=1920, num_codebooks=4, codebook_size=64, eos_code=64)
    def tokenize(self, text): ...                    # list[int]
    def speaker_embedding(self, pcm, sr): ...        # ndarray[D]
    def encode_codes(self, pcm, sr): ...             # ndarray[frames, num_codebooks] int16
    def begin(self, voice: dict, language): ...      # → ваше состояние сессии (любой объект)
    def append_text(self, st, tokens): ...
    def needs_text(self, st, text_ended): ...        # True, если для следующего кадра не хватает текста
    def talker_logits(self, st, text_ended): ...     # ndarray[codebook_size + 1], последний — EOS
    def accept_code0(self, st, code): ...
    def predictor_logits(self, st, cb): ...          # ndarray[codebook_size] для кодбука cb = 1..N-1
    def end_frame(self, st): ...
    def decode(self, st, codes): ...                 # codes[n, num_codebooks] → ndarray float32 PCM
    # необязательно: name, fingerprint, stats, tokenize_ref, text_safe_cut, accept_code, flush

class MyFamily(vox.Family):
    name = "my-model"                                # значение model.family в манифесте .tmod
    def footprint(self, manifest):                   # что нужно модели — для планировщика ресурсов
        return {"stages": [{"name": "model", "prefixes": ["w."], "session_ram_bytes": 1 << 20}]}
    def capabilities(self, manifest):
        return {"clone_icl": True, "clone_xvector": True, "languages": ["auto"]}
    def create(self, tmod, plan):                    # plan — принятый план ресурсов (dict)
        return MyBackend(tmod)

vox.register_family(MyFamily())
eng = vox.Engine("my-model.tmod")                    # дальше всё как с родными моделями
```

Что делает рантайм, а что — бэкенд:

| рантайм | бэкенд |
|---|---|
| сэмплинг кодов из логитов (top-k/top-p, температура, Philox: результат зависит только от seed) | логиты talker и predictor |
| подача текста частями, чанки, отмена, async-стриминг | токенизация, `needs_text` |
| лимиты RAM, число сессий, потоки | `footprint` (оценка памяти) |
| `.tvoice`, голоса, сервер | `speaker_embedding`, `encode_codes`, `begin` |

Исключение в Python-бэкенде возвращается как `vox.VoxError` с трассировкой, процесс не падает. Python-бэкенд считает
на CPU под GIL, поэтому подходит для прототипов, тестов и нетребовательных моделей; быстрый бэкенд пишется на C++
(`class Family` в `src/family.hpp`, образец — `make_test_family()` в `src/backend_null.cpp`).

`tests/test_py_family.py` — рабочий пример: тестовое семейство `vox-test`, переписанное на Python, даёт звук, побитово
совпадающий с C++-реализацией.

Границы: движок ожидает авторегрессионную схему «talker → кодбуки кадра → потоковый декодер»; модель «весь текст →
вся речь» этим интерфейсом не выражается.
