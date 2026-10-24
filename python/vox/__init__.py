"""vox — движок синтеза речи.

    import vox
    engine = vox.Engine("model.tmod")
    voice = engine.clone_voice("ref.wav", ref_text="Текст из референса", name="my_voice")
    voice.save("my_voice.tvoice")

    audio = engine.synthesize("Привет!", voice=voice, seed=42)
    vox.save_wav("out.wav", audio, engine.sample_rate)

    for chunk in engine.stream("Длинный текст…", voice=voice):
        player.write(chunk)
"""
from __future__ import annotations

import asyncio
import json
import os
import unicodedata
from typing import AsyncIterator, Iterator, Optional, Union

import numpy as np

from .audio import load_wav, save_wav, resample_linear  # noqa: F401

try:
    from . import _vox
except ImportError as _e:  # нативный модуль не собран: vox.formats и vox.audio всё равно работают
    _vox = None
    _import_error = _e

__version__ = _vox.VERSION if _vox else "0.19.2"
__all__ = ["Engine", "Voice", "Profile", "Session", "Duplex", "Resources", "VoxError", "OutOfMemory", "DeviceError", "devices", "inspect", "plan",
           "make_test_model", "register_family", "Family", "Backend", "Tmod", "load_wav", "save_wav", "__version__"]

AudioInput = Union[str, os.PathLike, "tuple[np.ndarray, int]"]


class VoxError(RuntimeError):
    """Ошибка движка. .status — код vox_status, .message — текст."""

    def __init__(self, status: int, message: str):
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message


class OutOfMemory(VoxError):
    """Не хватило памяти или лимит ресурсов (VOX_ERR_OOM): модель не открыта или сессия не принята."""


class DeviceError(VoxError):
    """Запрошенное устройство недоступно (VOX_ERR_DEVICE)."""


def _make_error(status: int, message: str) -> VoxError:
    return {-4: OutOfMemory, -3: DeviceError}.get(status, VoxError)(status, message)


if _vox:
    _vox.set_error_type(_make_error)

_PARAM_NAMES = ("temperature", "top_k", "top_p", "repetition_penalty", "sub_temperature",
                "sub_top_k", "sub_top_p", "seed", "chunk_frames", "max_frames", "language", "decoder_context_frames", "decoder_mode", "extra_json")


# параметры моделей без покадровых кодов (F5-TTS и др.): уходят в extra_json; можно и целиком словарём extra={...}
_EXTRA_NAMES = ("nfe_step", "cfg_strength", "sway_sampling_coef", "speed", "target_rms", "cross_fade_duration", "fix_duration",
                "phonemes", "lang", "style", "markup")
# profiles (0.15): voice_id = which voice of the profile, style = its style or an emotion, markup = "auto" | "strict" | "off"
_RENAMED = {"voice_id": "voice"}
# двусторонний стриминг: commit = "auto" | "sentence" | "manual", idle_ms — автокоммит после простоя (см. vox.h, ABI 0.4)
_STREAM_NAMES = ("commit", "idle_ms")


def _params(kw: dict) -> "_vox.Params":
    kw = dict(kw)
    extra = dict(kw.pop("extra", None) or {})
    for k in _EXTRA_NAMES:
        if k in kw:
            extra[k] = kw.pop(k)
    for k, to in _RENAMED.items():
        if k in kw:
            v = kw.pop(k)
            if v is not None:
                extra[to] = v
    stream = dict(extra.get("stream") or {})
    for k in _STREAM_NAMES:
        if k in kw:
            stream[k] = kw.pop(k)
    if stream:
        extra["stream"] = stream
    unknown = set(kw) - set(_PARAM_NAMES)
    if unknown:
        raise TypeError(f"unknown generation parameter(s): {', '.join(sorted(unknown))}")
    if extra:
        import json as _json
        kw["extra_json"] = _json.dumps(extra)
    return _vox.Params(**kw)


class Resources:
    """Сколько и каких ресурсов можно использовать. Все поля необязательны.

        vox.Resources(device="gpu:1", mode="memory", ram_limit="4GiB", vram_limit="2GiB",
                      threads=4, cpu_affinity="0-3", priority="low",
                      place={"codec": "cpu"}, gpu_layers={"talker": 12}, max_sessions=2)

    device: "auto" | "cpu" | "gpu" | "gpu:N" | "vulkan[:N]" (+ ",strict", ",codec=cpu" …)
    mode: "speed" | "balanced" | "memory" | "background" — по умолчанию подбирается по железу.
    ram_limit/vram_limit: байты или строка "6GiB", "512M". enforce: отказаться открывать модель, если план их превышает.
    """

    _KEYS = ("device", "mode", "gpu", "threads", "cpu_affinity", "priority", "ram_limit", "vram_limit", "place",
             "gpu_layers", "strict", "enforce", "overlap", "release_host_weights", "max_sessions", "planned_sessions", "tuned")

    def __init__(self, **kw):
        unknown = set(kw) - set(self._KEYS)
        if unknown:
            raise TypeError(f"unknown resource option(s): {', '.join(sorted(unknown))}; known: {', '.join(self._KEYS)}")
        self.options = {k: v for k, v in kw.items() if v is not None}

    @classmethod
    def coerce(cls, value) -> "Resources":
        if value is None:
            return cls()
        if isinstance(value, Resources):
            return value
        if isinstance(value, str):
            return cls(device=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("resources must be Resources, dict, device string or None")

    def to_json(self) -> str:
        return json.dumps(self.options)

    def __repr__(self) -> str:
        return "Resources(" + ", ".join(f"{k}={v!r}" for k, v in self.options.items()) + ")"


def devices() -> dict:
    """Процессор, память и GPU машины, плюс зарегистрированные семейства моделей. Ничего не загружает."""
    return json.loads(_native().devices_json())


def inspect(model: Union[str, os.PathLike]) -> dict:
    """Что за модель: семейство, вариант, стадии и их размер, возможности. Веса не читаются."""
    return json.loads(_native().inspect_json(os.fspath(model)))


def plan(model: Union[str, os.PathLike], resources=None) -> dict:
    """Куда будет размещена каждая стадия и сколько памяти потребуется при данных ресурсах. Веса не читаются.
    Ключи результата: feasible, backend, stages, ram_estimate, vram_estimate, notes, warnings, summary, …"""
    return json.loads(_native().plan_json(os.fspath(model), Resources.coerce(resources).to_json()))


def tune_info(model: Union[str, os.PathLike]) -> dict:
    """What is known about tuning this model on this computer: {"cache", "model_key", "hardware", "entry" (the saved result
    or None), "candidates" (configurations worth measuring)}. Reads no weights."""
    return json.loads(_native().tune_info_json(os.fspath(model)))


def tune(model: Union[str, os.PathLike], voice=None, **kw) -> dict:
    """Measure the candidate configurations of a model on this computer (each in its own process) and save the fastest:
    engines opened later with the default resources use it. See vox.tune.run for the options."""
    from .tune import run
    return run(model, voice, **kw)


def make_test_model(path: Union[str, os.PathLike]) -> str:
    """Записывает крошечную модель тестового семейства vox-test (для разработки и тестов без настоящих весов)."""
    _native().write_test_model(os.fspath(path))
    return os.fspath(path)


_NP_DTYPES = {"F32": "float32", "F16": "float16", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8"}


class Tmod:
    """Содержимое .tmod для Python-бэкенда: manifest (dict) и blob(name) → numpy-массив (читается из mmap без копии;
    bf16 отдаётся как uint16 — биты bfloat16)."""

    def __init__(self, view):
        self._v = view

    @property
    def manifest(self) -> dict:
        return self._v.manifest

    def blob_names(self) -> list:
        return self._v.blob_names()

    def blob(self, name: str) -> np.ndarray:
        raw, dt, shape = self._v.blob(name)
        np_dt = "uint16" if dt == "BF16" else _NP_DTYPES[dt]
        return raw.view(np_dt).reshape(shape)


class Family:
    """Базовый класс семейства моделей, написанного на Python. Подключается через vox.register_family(MyFamily()).

    Обязательно: name (str) и create(tmod: Tmod, plan: dict) -> Backend.
    Необязательно: matches(manifest) (по умолчанию manifest["model"]["family"] == name), footprint(manifest) -> dict
    (stages: [{name, weight_bytes, bytes_per_frame, session_ram_bytes}], always_ram_bytes), capabilities(manifest) -> dict.
    Бэкенд — объект с методами, перечисленными в vox.Backend (документация в docs/python-backends.md).
    Вычисления Python-бэкенда идут на CPU; всё остальное (сессии, потоковый текст, детерминированный сэмплинг, лимиты
    ресурсов, .tvoice, сервер) общее с родными моделями."""

    name = ""

    def create(self, tmod: Tmod, plan: dict):
        raise NotImplementedError


class Backend:
    """Контракт Python-бэкенда (все методы вызываются движком; тексты и аудио — numpy float32, коды — int16 [кадры, кодбуки]).

        info() -> dict: sample_rate, samples_per_frame, num_codebooks, codebook_size, eos_code, [suppress_from, suppress_to, min_eos_frames]
        tokenize(text) -> list[int]
        speaker_embedding(pcm, sr) -> ndarray[D]
        encode_codes(pcm, sr) -> ndarray[frames, num_codebooks] (int16)
        begin(voice: dict, language) -> state      # voice: name, ref_text, speaker_emb, ref_codes, ref_text_tokens, sampling, …
        append_text(state, tokens);  needs_text(state, text_ended) -> bool
        talker_logits(state, text_ended) -> ndarray[codebook_size + 1]   # последний — EOS
        accept_code0(state, code);   predictor_logits(state, cb) -> ndarray[codebook_size]
        accept_code(state, cb, code) [необязательно];  end_frame(state)
        decode(state, codes[n, num_codebooks]) -> ndarray pcm;  flush(state) -> ndarray [необязательно]
        name(), fingerprint(), stats(), tokenize_ref(text), text_safe_cut(pending_bytes) — необязательно
    """


def register_family(family) -> None:
    """Регистрирует семейство моделей (объект с name и create, см. vox.Family). Модели .tmod с model.family == name
    после этого открываются обычным vox.Engine(...), CLI-серверу и сессиям — без правок C++."""
    n = _native()
    if not getattr(family, "name", ""):
        raise ValueError("family.name is required")
    if not callable(getattr(family, "create", None)):
        raise ValueError("family.create(tmod, plan) is required")

    class _Adapter:
        name = family.name
        description = getattr(family, "description", "python-defined family")

        def create(self, view, plan):
            return family.create(Tmod(view), plan)

    adapter = _Adapter()
    for opt in ("matches", "footprint", "capabilities"):
        if hasattr(family, opt):
            setattr(adapter, opt, getattr(family, opt))
    n.register_family(adapter)


def _native():
    if _vox is None:
        raise ImportError(f"vox native module is not built: {_import_error}")
    return _vox


def _pcm(audio: AudioInput):
    if isinstance(audio, (str, os.PathLike)):
        pcm, sr = load_wav(audio)
    else:
        pcm, sr = audio
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


class Voice:
    """A voice profile (.tvoice): one or several voices, each with styles (emotions, delivery), the recordings any cloning model can
    rebuild it from, built-in speakers to use on models that cannot clone, and a cache of voices already built for concrete models.
    A profile from one model works on another: what is missing is built from the stored recording on first use (save() keeps it).

        p = vox.Profile("Cast")
        p.add_recording("anna", "anna.wav", "What Anna says in it")
        p.add_recording("anna", "anna_happy.wav", "Something cheerful", style="happy")
        p.set_style("anna", "whisper", instruct="Whisper softly", speed=0.95)
        p.set_voice("boris", speakers={"kokoro": "am_adam"})
        engine.synthesize("anna: Hi! [happy] Great news. [pause 300ms]\nboris: [whisper] Really?", p)
    """

    def __init__(self, handle: "_vox.Voice"):
        self._h = handle

    @classmethod
    def open(cls, path: Union[str, os.PathLike]) -> "Voice":
        """Open a .tvoice without a model (nothing is rebuilt): to edit it, merge it, or use it with any engine later."""
        return cls(_native().profile_open(os.fspath(path)))

    @property
    def info(self) -> dict:
        return json.loads(self._h.info())

    @property
    def name(self) -> str:
        return self.info["name"]

    @name.setter
    def name(self, value: str) -> None:
        self._h.set_name(value)

    @property
    def voices(self) -> dict:
        """{voice name: {description, language, default_style, styles: {…}, speakers: {…}, options: {…}}}"""
        return self.info.get("voices", {})

    @property
    def default_voice(self) -> str:
        return self.info.get("default_voice", "")

    def set_defaults(self, **params) -> "Voice":
        """Generation defaults of the default voice: seed, temperature, top_k, speed, …"""
        _vox.voice_set_defaults(self._h, _params(params))
        return self

    def add_recording(self, voice: str, audio: AudioInput, text: Optional[str] = None, *, style: Optional[str] = None,
                      language: Optional[str] = None) -> "Voice":
        """A reference recording of `voice` (created if new) for `style` (None = its default style). No model is needed."""
        pcm, sr = _pcm(audio)
        _vox.voice_add_recording(self._h, voice, style, pcm, sr, text, language)
        return self

    def add(self, other: "Voice", *, voice: Optional[str] = None, as_voice: Optional[str] = None, style: Optional[str] = None) -> "Voice":
        """Copy a voice of another profile (its recordings, settings and built caches). style=…: the other voice becomes one style of as_voice."""
        _vox.voice_merge(self._h, other._h, voice, as_voice, style)
        return self

    def update(self, patch: dict) -> "Voice":
        """JSON merge patch of the description (voices / styles / speakers / options / default_voice); None removes."""
        _vox.voice_update(self._h, json.dumps(patch))
        return self

    def set_voice(self, voice: str, *, description: Optional[str] = None, language: Optional[str] = None,
                  speakers: Optional[dict] = None, default_style: Optional[str] = None, default: bool = False, **options) -> "Voice":
        """Voice settings. speakers: {"kokoro": "af_heart", "qwen3-tts/custom_voice": "serena"} (None as a value removes one);
        options: generation defaults for the voice (speed, temperature, …)."""
        v = {}
        if description is not None:
            v["description"] = description
        if language is not None:
            v["language"] = language
        if speakers is not None:
            v["speakers"] = speakers
        if default_style is not None:
            v["default_style"] = default_style
        if options:
            v["options"] = options
        patch = {"voices": {voice: v}}
        if default:
            patch["default_voice"] = voice
        return self.update(patch)

    def set_style(self, voice: str, style: str, *, instruct: Optional[str] = None, **options) -> "Voice":
        """A style of a voice: an instruction (models that take one) and generation options (speed, …). Its recording comes from
        add_recording(…, style=…); without one the style uses the voice's default recording."""
        st = {}
        if instruct is not None:
            st["instruct"] = instruct
        if options:
            st["options"] = options
        return self.update({"voices": {voice: {"styles": {style: st}}}})

    def remove(self, voice: str, style: Optional[str] = None) -> "Voice":
        return self.update({"voices": {voice: None if style is None else {"styles": {style: None}}}})

    def prepare(self, engine: "Engine", voice: Optional[str] = None, style: Optional[str] = None) -> dict:
        """Build the voices (all, or one) for engine's model now; save() keeps them. Returns {"model", "voices", "notes"}."""
        return json.loads(_vox.voice_prepare(engine._h, self._h, voice, style))

    def save(self, path: Union[str, os.PathLike]) -> None:
        self._h.save(os.fspath(path))

    def __repr__(self) -> str:
        i = self.info
        vs = list(i.get("voices", {}))
        return f"<vox.Voice {i['name']!r} voices={vs} built={len(i.get('cache', []))}>"


class Profile(Voice):
    """A new, empty voice profile (vox.Profile("Cast")) — the same object as vox.Voice. Open a file with vox.Voice.open(path)."""

    def __init__(self, name: str = "profile"):
        super().__init__(_native().profile_new(name))


class Session:
    """Одна генерация. Текст можно подавать частями, аудио забирать по мере готовности.

        with engine.session(voice) as s:
            for piece in llm_stream:
                s.feed(piece)
                for chunk in s.ready():   # то, что уже готово, без ожидания
                    play(chunk)
            s.end()
            for chunk in s:               # остаток до конца
                play(chunk)
    """

    def __init__(self, handle: "_vox.Session", chunk_samples: int):
        self._h = handle
        self._chunk = chunk_samples
        self._ended = False

    def feed(self, text: str) -> "Session":
        # токенизатор Qwen нормализует текст в NFC; нативная часть этого не делает
        self._h.feed(unicodedata.normalize("NFC", text))
        return self

    def end(self) -> "Session":
        if not self._ended:
            self._h.end()
            self._ended = True
        return self

    def flush(self) -> "Session":
        """Передать модели всё накопленное (даже слово без пробела) и закрыть текущий кусок; сессия остаётся открытой."""
        if not self._ended:
            self._h.flush()
        return self

    def set_pending(self, text: str) -> "Session":
        """Заменить ещё не переданный модели хвост текста (правка при наборе). Уже переданное назад не берётся."""
        self._h.set_pending(unicodedata.normalize("NFC", text))
        return self

    @property
    def info(self) -> dict:
        """Состояние текста: committed_bytes, pending, text_ended, commit, flushes… (vox_session_info)."""
        return json.loads(self._h.info())

    def cancel(self) -> None:
        self._h.cancel()

    def ready(self) -> Iterator[np.ndarray]:
        """Чанки, готовые прямо сейчас (останавливается, когда нужен ещё текст)."""
        while True:
            st, chunk = self._h.pull(self._chunk)
            if st == _vox.OK:
                yield chunk
            else:
                return

    def __iter__(self) -> Iterator[np.ndarray]:
        self.end()
        while True:
            st, chunk = self._h.pull(self._chunk)
            if st == _vox.OK:
                yield chunk
            elif st == _vox.DONE:
                return
            else:  # AGAIN после end() быть не должно
                raise VoxError(st, "session returned AGAIN after end of text")

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.cancel()


class _KokoroSession(Session):
    """Session whose text goes through misaki: complete lines become phoneme pieces (one per line for the engine)."""

    def __init__(self, handle, chunk_samples: int, frontend, names=()):
        super().__init__(handle, chunk_samples)
        self._fe = frontend
        self._buf = ""
        self._names = {n.lower() for n in names}

    _TAG = None

    def _push(self, text: str) -> None:
        # markup ([happy], [pause 300ms], "anna: " at a line start) passes through as is; only the text between it is phonemized
        import re
        if _KokoroSession._TAG is None:
            _KokoroSession._TAG = re.compile(r"(\\\[|\[[^\[\]\n]{1,64}\]|(?m:^[ \t]*[^\s:\[\]]{1,64}:(?=\s)))")
        out = []
        for part in _KokoroSession._TAG.split(text):
            if not part:
                continue
            if _KokoroSession._TAG.fullmatch(part):
                name = part.strip()[:-1] if part.rstrip().endswith(":") else None
                if name is not None and name.lower() not in self._names:
                    pieces = self._fe.chunks(part)
                    if pieces:
                        out.append("\n".join(pieces) + "\n")
                    continue
                out.append(part.strip() if name is not None else part)
            elif part.strip():
                pieces = self._fe.chunks(part)
                if pieces:
                    out.append("\n".join(pieces) + "\n")
            elif "\n" in part:
                out.append("\n")
        if out:
            self._h.feed("".join(out))

    def feed(self, text: str) -> "Session":
        self._buf += unicodedata.normalize("NFC", text)
        if "\n" in self._buf:
            done, self._buf = self._buf.rsplit("\n", 1)
            self._push(done)
        return self

    def end(self) -> "Session":
        if not self._ended and self._buf.strip():
            self._push(self._buf)
        self._buf = ""
        return super().end()

    def flush(self) -> "Session":
        if self._buf.strip():
            self._push(self._buf)
        self._buf = ""
        return super().flush()


class Engine:
    """Загруженная модель (.tmod)."""

    def __init__(self, path: Union[str, os.PathLike], device: str = "auto", threads: int = 0, *, resources=None):
        """device/threads — короткий способ; resources (Resources, dict или строка) — полный набор ограничений
        и перекрывает device/threads."""
        _native()
        res = Resources.coerce(resources)
        self._h = _vox.Engine(os.fspath(path), device, threads, res.to_json() if res.options else "")
        self.resources = res

    def close(self) -> None:
        """Освобождает модель сразу (иначе — когда объект соберёт сборщик мусора)."""
        self._h = None

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def sample_rate(self) -> int:
        return self._h.sample_rate

    @property
    def info(self) -> dict:
        return json.loads(self._h.info())

    @property
    def plan(self) -> dict:
        """Принятый план ресурсов: где считается каждая стадия, оценка памяти, предупреждения."""
        return self.info["plan"]

    @property
    def capabilities(self) -> dict:
        """Что умеет модель: clone_icl, clone_xvector, named_speakers, instruct, speakers, languages, …"""
        return self.info["capabilities"]

    @property
    def speakers(self) -> list:
        return list(self.capabilities.get("speakers", []))

    @property
    def languages(self) -> list:
        return list(self.capabilities.get("languages", []))

    def stats(self) -> dict:
        """Текущая статистика: память (RSS, бюджет, VRAM), время стадий, сессии, пропускная способность."""
        return json.loads(self._h.stats())

    # ---- голоса ----
    def clone_voice(self, audio: AudioInput, ref_text: Optional[str] = None, *, name: Optional[str] = None,
                    xvector_only: bool = False, **defaults) -> Voice:
        """Создаёт профиль голоса из записи: путь к WAV или (массив, частота)."""
        if isinstance(audio, (str, os.PathLike)):
            pcm, sr = load_wav(audio)
        else:
            pcm, sr = audio
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        if pcm.ndim == 2:
            pcm = np.ascontiguousarray(pcm.mean(axis=1), dtype=np.float32)
        v = Voice(_vox.voice_create(self._h, pcm, int(sr), ref_text, xvector_only))
        if name:
            v.name = name
        if defaults:
            v.set_defaults(**defaults)
        return v

    def speaker_voice(self, speaker: Optional[str] = None, instruct: Optional[str] = None, *, name: Optional[str] = None,
                      **defaults) -> Voice:
        """Голос без записи: готовый диктор модели CustomVoice (имя из engine.speakers, регистр не важен) и/или инструкция
        о стиле (CustomVoice 1.7B). Для VoiceDesign используйте design_voice."""
        v = Voice(_vox.voice_from_speaker(self._h, speaker, instruct))
        if name:
            v.name = name
        if defaults:
            v.set_defaults(**defaults)
        return v

    def design_voice(self, instruct: str, *, name: Optional[str] = None, **defaults) -> Voice:
        """Голос по текстовому описанию (модель VoiceDesign): «A deep calm male voice»."""
        return self.speaker_voice(None, instruct, name=name, **defaults)

    def load_voice(self, path: Union[str, os.PathLike]) -> Voice:
        return Voice(_vox.voice_load(self._h, os.fspath(path)))

    def _voice(self, voice: Union[Voice, str, os.PathLike]) -> Voice:
        return voice if isinstance(voice, Voice) else self.load_voice(voice)

    # ---- синтез ----
    def session(self, voice: Union[Voice, str, os.PathLike], *, chunk_samples: int = 8192, g2p: Optional[str] = None,
                **params) -> Session:
        """g2p (Kokoro): "misaki" — text → phonemes with misaki (+ espeak-ng), exactly like kokoro.KPipeline; "builtin" — the
        engine's own G2P; None — misaki when it is installed. phonemes=True: the text already is phonemes."""
        v = self._voice(voice)
        if self.info.get("family") == "kokoro" and not params.get("phonemes"):
            from . import kokoro_frontend as kf
            if g2p == "misaki" or (g2p is None and kf.available()):
                info = v.info
                spk = info.get("speaker") or v.name
                vid = params.get("voice_id") or info.get("default_voice")
                for key in (f"kokoro/{self.info['manifest']['model'].get('variant', '')}", "kokoro"):
                    spk = (info.get("voices", {}).get(vid, {}).get("speakers", {}) or {}).get(key) or spk
                fe = kf.frontend(kf.is_british(spk, params.get("lang")))
                params = dict(params, phonemes=True)
                names = list(info.get("voices", {})) if len(info.get("voices", {})) > 1 else []
                return _KokoroSession(_vox.session_create(self._h, v._h, _params(params)), chunk_samples, fe, names)
            if g2p not in (None, "builtin"):
                raise ValueError('g2p must be "misaki", "builtin" or None')
        return Session(_vox.session_create(self._h, v._h, _params(params)), chunk_samples)

    def phonemize(self, text: str, lang: Optional[str] = None, *, g2p: Optional[str] = None) -> dict:
        """Kokoro: text → phonemes, {"phonemes": str, "chunks": [str, ...]} (chunks are what gets synthesized).
        g2p as in session(): misaki when installed, else the built-in G2P."""
        if self.info.get("family") == "kokoro" and g2p != "builtin":
            from . import kokoro_frontend as kf
            if g2p == "misaki" or kf.available():
                fe = kf.frontend(kf.is_british(None, lang))
                chunks = fe.chunks(text)
                return {"phonemes": "\n".join(chunks), "chunks": chunks, "g2p": "misaki"}
        r = json.loads(self._h.phonemize(text, lang or ""))
        r["g2p"] = "builtin"
        return r

    def duplex(self, voice: Union[Voice, str, os.PathLike], **kw) -> "Duplex":
        """Двусторонний стрим: пишете текст в любой момент, звук приходит по мере готовности. См. vox.Duplex."""
        from .duplex import Duplex
        return Duplex(self, self._voice(voice), **kw)

    def stream(self, text: str, voice: Union[Voice, str, os.PathLike], **params) -> Iterator[np.ndarray]:
        s = self.session(voice, **params)
        s.feed(text)
        yield from s

    def synthesize(self, text: str, voice: Union[Voice, str, os.PathLike], **params) -> np.ndarray:
        chunks = list(self.stream(text, voice, **params))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def synthesize_many(self, texts, voice: Union[Voice, str, os.PathLike], *, workers: Optional[int] = None,
                        **params) -> "list[np.ndarray]":
        """Синтез нескольких текстов параллельно сессиями одного движка. Число одновременных сессий ограничено
        workers (по умолчанию — max_sessions из ресурсов или 2). seed, если задан, одинаков для всех текстов;
        результат каждого текста не зависит от остальных."""
        from concurrent.futures import ThreadPoolExecutor
        texts = list(texts)
        v = self._voice(voice)
        n = workers or self.resources.options.get("max_sessions") or 2
        with ThreadPoolExecutor(max_workers=max(1, min(int(n), len(texts) or 1))) as ex:
            return list(ex.map(lambda t: self.synthesize(t, v, **params), texts))

    def benchmark(self, text: str, voice: Union[Voice, str, os.PathLike], *, runs: int = 3, **params) -> dict:
        """Замер скорости через библиотеку (без CLI): первый чанк, RTF, секунды речи. Первый прогон считается холодным."""
        import time
        v = self._voice(voice)
        rows = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            first, samples = None, 0
            for chunk in self.stream(text, v, **params):
                if first is None:
                    first = time.perf_counter() - t0
                samples += len(chunk)
            total = time.perf_counter() - t0
            audio = samples / self.sample_rate
            rows.append({"first_chunk_s": first, "total_s": total, "audio_s": audio, "rtf": total / audio if audio else None})
        warm = rows[1:] or rows
        mean = lambda k: sum(r[k] for r in warm if r[k] is not None) / max(1, len(warm))
        return {"runs": rows, "warm": {"first_chunk_s": mean("first_chunk_s"), "total_s": mean("total_s"),
                                       "audio_s": mean("audio_s"), "rtf": mean("rtf")}, "stats": self.stats()}

    async def astream(self, text: str, voice: Union[Voice, str, os.PathLike], **params) -> AsyncIterator[np.ndarray]:
        """Асинхронный стрим: генерация идёт в пуле потоков, event loop не блокируется."""
        it = self.stream(text, voice, **params)
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, it, sentinel)
            if chunk is sentinel:
                return
            yield chunk


def __getattr__(name):  # vox.Duplex импортируется лениво (потоки не нужны тем, кто их не использует)
    if name == "Duplex":
        from .duplex import Duplex
        return Duplex
    raise AttributeError(name)
