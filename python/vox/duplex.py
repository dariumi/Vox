"""Двусторонний стриминг: текст подаётся на лету, звук забирается на лету — из разных потоков или корутин.

    with engine.duplex(voice, commit="auto", idle_ms=600) as d:
        d.write("Привет, ")            # откуда угодно и когда угодно (например, токены из LLM)
        d.write("как дела?")
        d.end()                        # текст закончился
        for chunk in d.audio():        # блокирующий итератор; идёт, пока не закончится звук
            play(chunk)

Режимы (см. и «Режимы» в README):
  * живой вход → живой выход:       write(...) в одном потоке, for chunk in d.audio() в другом;
  * живой вход → одна запись:       d = engine.duplex(voice, commit="manual"); write(...); end(); audio = d.final();
  * набор текста с правками:        d.set_text(текст_в_поле_ввода) при каждом изменении — озвучивается набранное, правки
                                    ещё не озвученного хвоста учитываются, уже озвученное не переозвучивается (см. on_conflict);
  * весь текст → потоковый звук:    d.write(text); d.end(); for chunk in d.audio(): ...

Что и когда уходит в модель, определяет commit (auto | sentence | manual) и idle_ms (автокоммит после простоя) — это политики
движка (vox.h). Модели разные: Qwen3-TTS читает текст по словам, F5-TTS озвучивает куски целиком (engine.capabilities["stream"]).
utterance_gap_ms > 0: после такой паузы без текста высказывание закрывается (модель дочитывает хвост), а следующий write()
открывает новое — так диктовка не зависает на последних словах (AR-модели без конца текста не знают, что фраза закончилась).
"""
from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
import unicodedata
from typing import Iterator, Optional

import numpy as np

from . import VoxError, _vox


class Duplex:
    def __init__(self, engine, voice, *, commit: str = "auto", idle_ms: int = 0, utterance_gap_ms: int = 0,
                 on_conflict: str = "keep", chunk_samples: int = 4096, seed: Optional[int] = None, **params):
        if on_conflict not in ("keep", "raise", "respeak"):
            raise ValueError("on_conflict: keep | raise | respeak")
        if commit not in ("auto", "sentence", "manual"):
            raise ValueError("commit: auto | sentence | manual")
        self._engine, self._voice = engine, voice
        self._params = dict(params, commit=commit, idle_ms=int(idle_ms))
        self._seed = seed if seed is not None else int(params.pop("seed", 0) or 1)
        self._params.pop("seed", None)
        self._gap = utterance_gap_ms / 1000.0
        self._on_conflict = on_conflict
        self._chunk = chunk_samples
        self._cmds: "queue.Queue[tuple]" = queue.Queue()
        self._out: "queue.Queue[object]" = queue.Queue()
        self._done = threading.Event()
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._ended = False
        self.utterances = 0
        self.stats = {"first_audio_s": None, "audio_samples": 0, "committed_bytes": 0}
        self._t_first_write: Optional[float] = None
        self._thread = threading.Thread(target=self._run, name="vox-duplex", daemon=True)
        self._thread.start()

    # ---------------- вход ----------------
    def _send(self, kind, arg=None, wait=False):
        if self._error:
            raise self._error
        fut = concurrent.futures.Future()
        self._cmds.put((kind, arg, fut))
        return fut.result() if wait else fut

    def write(self, text: str) -> "Duplex":
        """Дописать текст. Не блокирует (команда уходит рабочему потоку)."""
        if self._ended:
            raise VoxError(-7, "text already ended")
        if text:
            if self._t_first_write is None:
                self._t_first_write = time.perf_counter()
            self._send("write", unicodedata.normalize("NFC", text))
        return self

    def flush(self) -> "Duplex":
        """Передать модели всё накопленное и закрыть текущий кусок (например, по паузе в речи или по Enter)."""
        self._send("flush")
        return self

    def set_text(self, full_text: str) -> dict:
        """Режим набора: передаём ВЕСЬ текст из поля ввода. Дописанное и правки ещё не озвученного хвоста применяются; правка
        внутри уже озвученного (переданного модели) разрешается по on_conflict: keep — игнорировать, raise — исключение,
        respeak — закрыть высказывание и озвучить текст, начиная с изменённого слова, как новое.
        Возвращает {"ok", "committed", "pending"}; блокирует до обработки (не дольше одного шага модели)."""
        if self._ended:
            raise VoxError(-7, "text already ended")
        if self._t_first_write is None and full_text:
            self._t_first_write = time.perf_counter()
        res = self._send("typing", unicodedata.normalize("NFC", full_text), wait=True)
        if not res["ok"] and self._on_conflict == "raise":
            raise ValueError(f"edit inside already spoken text (spoken: {res['spoken']!r})")
        return res

    def end(self) -> "Duplex":
        """Текст закончился: модель дочитает и закроет звук."""
        if not self._ended:
            self._ended = True
            self._send("end")
        return self

    def cancel(self) -> None:
        self._ended = True
        self._send("cancel")

    # ---------------- выход ----------------
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Один кусок звука; None — звук закончился (или timeout вышел: различайте по done)."""
        try:
            item = self._out.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            self._out.put(None)  # маркер конца остаётся для повторных вызовов
            if self._error:
                raise self._error
            return None
        return item

    def audio(self) -> Iterator[np.ndarray]:
        """Блокирующий итератор по кускам звука до конца (после end() или cancel())."""
        while True:
            item = self._out.get()
            if item is None:
                self._out.put(None)
                if self._error:
                    raise self._error
                return
            yield item

    async def aaudio(self):
        """То же для asyncio: ожидание идёт в пуле потоков."""
        import asyncio
        while True:
            item = await asyncio.to_thread(self._out.get)
            if item is None:
                self._out.put(None)
                if self._error:
                    raise self._error
                return
            yield item

    def final(self, timeout: Optional[float] = None) -> np.ndarray:
        """Закрывает текст (если ещё не закрыт), ждёт и возвращает всю запись одним массивом."""
        self.end()
        parts = []
        t0 = time.perf_counter()
        while True:
            left = None if timeout is None else max(0.0, timeout - (time.perf_counter() - t0))
            try:
                item = self._out.get(timeout=left)
            except queue.Empty:
                raise TimeoutError("duplex: audio is not finished yet")
            if item is None:
                self._out.put(None)
                break
            parts.append(item)
        if self._error:
            raise self._error
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def info(self) -> dict:
        """Состояние текста в движке (committed_bytes, pending, flushes…) — снимок рабочего потока."""
        return self._send("info", wait=True)

    def close(self) -> None:
        if not self._done.is_set():
            self.cancel()
        self._thread.join(timeout=5)

    def __enter__(self) -> "Duplex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or not self._done.is_set():
            self.cancel()
        self._thread.join(timeout=5)

    # ---------------- рабочий поток ----------------
    def _open(self):
        self.utterances += 1
        p = dict(self._params, seed=self._seed + self.utterances - 1)
        return self._engine.session(self._voice, chunk_samples=self._chunk, **p)

    def _run(self):
        s = None
        logical = b""              # текст, как он выглядит для пользователя в текущем высказывании (UTF-8)
        last_text = time.perf_counter()
        ended = False
        try:
            while True:
                # 1) команды
                try:
                    cmd = self._cmds.get(timeout=0.004)
                except queue.Empty:
                    cmd = None
                while cmd is not None:
                    kind, arg, fut = cmd
                    try:
                        if kind == "cancel":
                            if s is not None:
                                s.cancel()
                            fut.set_result(None)
                            return
                        if kind == "write":
                            if s is None:
                                s, logical = self._open(), b""
                            s.feed(arg)
                            logical += arg.encode()
                            last_text = time.perf_counter()
                            fut.set_result(None)
                        elif kind == "flush":
                            if s is not None:
                                s.flush()
                            fut.set_result(None)
                        elif kind == "info":
                            fut.set_result(s.info if s is not None else {"committed_bytes": 0, "pending": "", "text_ended": ended, "utterances": self.utterances})
                        elif kind == "typing":
                            s, logical, res = self._typing(s, logical, arg)
                            last_text = time.perf_counter()
                            fut.set_result(res)
                        elif kind == "end":
                            ended = True
                            if s is None:
                                s, logical = self._open(), b""
                            s.end()
                            fut.set_result(None)
                    except BaseException as e:  # noqa: BLE001 — ошибка команды уходит вызывающему
                        fut.set_exception(e)
                        if kind in ("write", "end"):
                            raise
                    try:
                        cmd = self._cmds.get_nowait()
                    except queue.Empty:
                        cmd = None
                # 2) звук
                if s is None:
                    continue
                while True:
                    st, chunk = s._h.pull(self._chunk)
                    if st == _vox.OK:
                        if self.stats["first_audio_s"] is None and self._t_first_write is not None:
                            self.stats["first_audio_s"] = time.perf_counter() - self._t_first_write
                        self.stats["audio_samples"] += len(chunk)
                        self._out.put(chunk)
                        continue
                    break
                if st == _vox.DONE:
                    self.stats["committed_bytes"] += s.info.get("committed_bytes", 0)
                    s = None
                    if ended:
                        return
                    continue
                # 3) пауза между высказываниями: закрываем текущее, следующий write откроет новое
                if self._gap and not ended and time.perf_counter() - last_text >= self._gap and s.info.get("committed_bytes", 0) + len(s.info.get("pending", "")) > 0:
                    s.end()
                    last_text = time.perf_counter() + 1e9  # не повторять, пока не придёт новый текст
        except BaseException as e:  # noqa: BLE001
            self._error = e
        finally:
            self._done.set()
            self._out.put(None)
            # команды, пришедшие после завершения, не должны зависнуть
            while True:
                try:
                    _, _, fut = self._cmds.get_nowait()
                except queue.Empty:
                    break
                if not fut.done():
                    fut.set_exception(self._error or VoxError(-7, "duplex session is closed"))

    def _typing(self, s, logical: bytes, full: str):
        new = full.encode()
        if s is None:
            s = self._open()
            logical = b""
        committed = s.info["committed_bytes"]
        if new[:committed] == logical[:committed] and len(logical) >= committed:
            s.set_pending(new[committed:].decode())
            return s, new, {"ok": True, "committed": committed, "pending": new[committed:].decode()}
        # правка внутри уже переданного модели текста
        spoken = logical[:committed].decode(errors="replace")
        res = {"ok": False, "committed": committed, "spoken": spoken, "pending": logical[committed:].decode(errors="replace")}
        if self._on_conflict == "respeak":
            i = 0
            while i < min(len(new), committed) and new[i] == logical[i]:
                i += 1
            # с начала изменённого слова
            j = new.rfind(b" ", 0, i) + 1
            s.end()                                     # старое высказывание дочитывается как есть
            self._finish_now(s)
            s = self._open()
            logical = new[j:]
            s.feed(new[j:].decode())
            res.update(ok=True, respeak_from=j, committed=0, pending=s.info["pending"])
            return s, logical, res
        return s, logical, res

    def _finish_now(self, s):
        """Дочитывает аудио завершаемого высказывания в выходную очередь (до DONE)."""
        while True:
            st, chunk = s._h.pull(self._chunk)
            if st == _vox.OK:
                self.stats["audio_samples"] += len(chunk)
                self._out.put(chunk)
            elif st == _vox.DONE:
                return
            else:
                raise VoxError(-7, "session returned AGAIN after end of text")
