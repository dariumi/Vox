"""WAV без внешних зависимостей (только numpy): PCM 16/24/32 и float32."""
from __future__ import annotations

import os
import struct
from typing import Tuple, Union

import numpy as np


def load_wav(path: Union[str, os.PathLike]) -> Tuple[np.ndarray, int]:
    """Возвращает (mono float32, частота). path — путь, bytes или файловый объект."""
    if isinstance(path, (bytes, bytearray)):
        data = bytes(path)
    elif hasattr(path, "read"):
        data = path.read()
    else:
        data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    fmt = ch = bits = sr = None
    pos = 12
    payload = None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, ch, sr = struct.unpack_from("<HHI", body, 0)
            bits = struct.unpack_from("<H", body, 14)[0]
            if fmt == 0xFFFE and size >= 26:
                fmt = struct.unpack_from("<H", body, 24)[0]
        elif cid == b"data":
            payload = body
            break
        pos += 8 + size + (size & 1)
    if payload is None or fmt is None:
        raise ValueError(f"{path}: WAV without fmt/data chunk")

    if fmt == 3 and bits == 32:
        x = np.frombuffer(payload, dtype="<f4")
    elif fmt == 1 and bits == 16:
        x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif fmt == 1 and bits == 32:
        x = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    elif fmt == 1 and bits == 24:
        b = np.frombuffer(payload[: len(payload) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = (b[:, 0] << 8) | (b[:, 1] << 16) | (b[:, 2] << 24)
        x = (v >> 8).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"{path}: unsupported WAV encoding (format {fmt}, {bits} bit)")
    n = len(x) // ch * ch
    x = x[:n].reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def save_wav(path: Union[str, os.PathLike], audio: np.ndarray, sample_rate: int, *, float32: bool = False) -> None:
    """Пишет mono WAV: PCM16 по умолчанию, float32 при float32=True."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if float32:
        fmt, bits, payload = 3, 32, a.astype("<f4").tobytes()
    else:
        fmt, bits = 1, 16
        payload = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(payload), b"WAVE", b"fmt ", 16, fmt, 1,
                      sample_rate, sample_rate * bits // 8, bits // 8, bits, b"data", len(payload))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(payload)


def resample_linear(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Простая линейная передискретизация (для тестов и предпросмотра, не для эталона)."""
    if sr_from == sr_to:
        return np.asarray(audio, dtype=np.float32)
    n = int(round(len(audio) * sr_to / sr_from))
    t = np.linspace(0, len(audio) - 1, n)
    return np.interp(t, np.arange(len(audio)), audio).astype(np.float32)
