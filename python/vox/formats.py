"""Чтение и запись .tvoice и .tmod на чистом Python (numpy), без нативной библиотеки.

Нужен инструментам: экспорт голосов из оригинального пайплайна, упаковка весов,
проверка файлов. Форматы описаны в спецификации; эталонная реализация — src/formats.cpp.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8}
_DT_REV = {np.dtype(v).name: k for k, v in _DT.items()}

TMOD_MAGIC = b"VOXM"
TMOD_ALIGN = 4096
TVOICE_FORMAT = "1.0"


def _dtype_name(a: np.ndarray) -> str:
    try:
        return _DT_REV[a.dtype.name]
    except KeyError:
        raise TypeError(f"unsupported dtype {a.dtype} (use float32/float16/int32/int16/int8/uint8)")


# ---------------- safetensors ----------------
def save_safetensors(path: str, tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None) -> None:
    header, off, blobs = {}, 0, []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": _dtype_name(a), "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def load_safetensors(path: str):
    data = open(path, "rb").read()
    n = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8:8 + n])
    meta = header.pop("__metadata__", {})
    body = memoryview(data)[8 + n:]
    out = {}
    for name, t in header.items():
        b, e = t["data_offsets"]
        out[name] = np.frombuffer(body[b:e], dtype=np.dtype(_DT[t["dtype"]]).newbyteorder("<")).reshape(t["shape"]).copy()
    return out, meta


# ---------------- .tvoice ----------------
@dataclass
class VoiceProfile:
    speaker_emb: np.ndarray                       # f32 [D]
    name: str = "voice"
    mode: str = "icl"                             # "icl" | "xvector_only"
    ref_text: str = ""
    language: str = "auto"
    ref_codes: Optional[np.ndarray] = None        # i16 [frames, codebooks]
    ref_text_tokens: Optional[np.ndarray] = None  # i32 [N]
    ref_audio: Optional[bytes] = None             # исходная запись (WAV)
    model: dict = field(default_factory=lambda: {"family": "", "variant": "", "weights_sha256": ""})
    sampling: dict = field(default_factory=dict)
    kv_cache: Dict[str, np.ndarray] = field(default_factory=dict)  # "kv.<fingerprint>.<name>"
    created: str = ""

    def save(self, path: str) -> None:
        tensors = {"speaker_emb": np.asarray(self.speaker_emb, dtype=np.float32).reshape(-1)}
        if self.mode == "icl":
            if self.ref_codes is None or self.ref_text_tokens is None:
                raise ValueError("ICL voice needs ref_codes and ref_text_tokens")
            codes = np.asarray(self.ref_codes)
            if codes.ndim != 2:
                raise ValueError("ref_codes must be [frames, codebooks]")
            tensors["ref_codes"] = codes.astype(np.int16)
            tensors["ref_text_tokens"] = np.asarray(self.ref_text_tokens, dtype=np.int32).reshape(-1)
        if self.ref_audio:
            tensors["ref_audio"] = np.frombuffer(self.ref_audio, dtype=np.uint8)
        tensors.update(self.kv_cache)
        fps = sorted({k.split(".")[1] for k in self.kv_cache if k.startswith("kv.") and k.count(".") >= 2})
        rc = tensors.get("ref_codes")
        meta = {
            "format": TVOICE_FORMAT, "name": self.name, "mode": self.mode, "ref_text": self.ref_text,
            "language": self.language, "model": self.model, "sampling": self.sampling, "kv_fingerprints": fps,
            "speaker_emb_dim": int(tensors["speaker_emb"].shape[0]),
            "ref_frames": int(rc.shape[0]) if rc is not None else 0,
            "ref_codebooks": int(rc.shape[1]) if rc is not None else 0,
            "has_ref_audio": bool(self.ref_audio),
            "created": self.created or _dt.date.today().isoformat(),
        }
        save_safetensors(path, tensors, {"format": TVOICE_FORMAT, "voice": json.dumps(meta, ensure_ascii=False)})

    @classmethod
    def load(cls, path: str) -> "VoiceProfile":
        t, meta = load_safetensors(path)
        if "voice" not in meta:
            raise ValueError(f"{path}: not a .tvoice file")
        j = json.loads(meta["voice"])
        return cls(
            speaker_emb=t["speaker_emb"], name=j.get("name", "voice"), mode=j.get("mode", "icl"),
            ref_text=j.get("ref_text", ""), language=j.get("language", "auto"),
            ref_codes=t.get("ref_codes"), ref_text_tokens=t.get("ref_text_tokens"),
            ref_audio=t["ref_audio"].tobytes() if "ref_audio" in t else None,
            model=j.get("model", {}), sampling=j.get("sampling", {}),
            kv_cache={k: v for k, v in t.items() if k.startswith("kv.")}, created=j.get("created", ""),
        )


# ---------------- .tmod ----------------
class LazyBlob:
    """Блоб, чьи данные вычисляются при записи (чтобы не держать всю модель в памяти)."""

    def __init__(self, dtype: str, shape, producer):
        self.dtype, self.shape, self.producer = dtype, list(shape), producer
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1, "U8": 1}[dtype]

    def tobytes(self) -> bytes:
        b = self.producer()
        if isinstance(b, np.ndarray):
            b = np.ascontiguousarray(b).tobytes()
        assert len(b) == self.nbytes, f"blob size mismatch: {len(b)} != {self.nbytes}"
        return b


def write_tmod(path: str, manifest: dict, blobs: Dict[str, np.ndarray], specialized: bool = False) -> None:
    """Пишет .tmod: заголовок 64 байта, JSON-манифест, блобы с выравниванием 4096."""
    table, off, arrays = {}, 0, []
    for name in sorted(blobs):
        a = blobs[name]
        if not isinstance(a, LazyBlob):
            a = np.ascontiguousarray(a)
            a = LazyBlob(_dtype_name(a), a.shape, (lambda x=a: x))
        off = -(-off // TMOD_ALIGN) * TMOD_ALIGN
        table[name] = {"dtype": a.dtype, "shape": a.shape, "offset": off, "nbytes": a.nbytes}
        arrays.append((off, a))
        off += a.nbytes
    m = dict(manifest)
    m["blobs"] = table
    mj = json.dumps(m, ensure_ascii=False, indent=2).encode()
    data_off = -(-(64 + len(mj)) // TMOD_ALIGN) * TMOD_ALIGN
    hdr = struct.pack("<4sHHIIQQQQ16s", TMOD_MAGIC, 1, 0, 1 if specialized else 0, 0,
                      64, len(mj), data_off, off, b"\0" * 16)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(mj)
        f.write(b"\0" * (data_off - 64 - len(mj)))
        pos = 0
        for o, a in arrays:
            f.write(b"\0" * (o - pos))
            f.write(a.tobytes())
            pos = o + a.nbytes


def read_tmod_manifest(path: str) -> dict:
    with open(path, "rb") as f:
        hdr = f.read(64)
        magic, major, minor, flags, _, moff, msize, doff, dsize, _ = struct.unpack("<4sHHIIQQQQ16s", hdr)
        if magic != TMOD_MAGIC:
            raise ValueError(f"{path}: not a .tmod file")
        f.seek(moff)
        m = json.loads(f.read(msize))
    m["_header"] = {"version": f"{major}.{minor}", "specialized": bool(flags & 1), "data_size": dsize}
    return m
