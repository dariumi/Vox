"""Минимальный WebSocket (RFC 6455) на стандартной библиотеке — для /v1/duplex сервера vox. Только то, что нужно:
рукопожатие, текстовые/бинарные кадры, фрагментация, ping/pong, close. Без расширений (permessage-deflate не предлагается)."""
from __future__ import annotations

import base64
import hashlib
import struct
import threading

_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1(key.encode() + _GUID).digest()).decode()


def encode_frame(op: int, payload: bytes, mask: bytes | None = None) -> bytes:
    n = len(payload)
    head = bytes([0x80 | op])
    m = 0x80 if mask else 0
    if n < 126:
        head += bytes([m | n])
    elif n < 65536:
        head += bytes([m | 126]) + struct.pack(">H", n)
    else:
        head += bytes([m | 127]) + struct.pack(">Q", n)
    if mask:
        payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        return head + mask + payload
    return head + payload


def _read_exact(f, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf += chunk
    return buf


class Closed(Exception):
    pass


class WebSocket:
    """Серверная сторона: читает кадры клиента (с маской), пишет без маски. send_* потокобезопасны."""

    def __init__(self, rfile, wfile, max_message: int = 1 << 22, client: bool = False):
        self.r, self.w = rfile, wfile
        self.max = max_message
        self._lock = threading.Lock()
        self.closed = False
        self._client = client

    def _write(self, op: int, payload: bytes):
        with self._lock:
            if self.closed and op != OP_CLOSE:
                raise Closed()
            mask = None
            if self._client:
                import os
                mask = os.urandom(4)
            self.w.write(encode_frame(op, payload, mask))
            self.w.flush()

    def send_text(self, s: str):
        self._write(OP_TEXT, s.encode())

    def send_binary(self, b: bytes):
        self._write(OP_BIN, b)

    def close(self, code: int = 1000):
        if not self.closed:
            try:
                self._write(OP_CLOSE, struct.pack(">H", code))
            except Exception:  # noqa: BLE001
                pass
            self.closed = True

    def recv(self):
        """→ ("text", str) | ("binary", bytes); поднимает Closed, когда клиент закрыл соединение."""
        msg, kind = b"", None
        while True:
            try:
                b0, b1 = _read_exact(self.r, 2)
            except EOFError:
                self.closed = True
                raise Closed()
            fin, op = b0 & 0x80, b0 & 0x0F
            masked, n = b1 & 0x80, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", _read_exact(self.r, 2))[0]
            elif n == 127:
                n = struct.unpack(">Q", _read_exact(self.r, 8))[0]
            if n > self.max:
                self.close(1009)
                raise Closed()
            mask = _read_exact(self.r, 4) if masked else None
            data = _read_exact(self.r, n) if n else b""
            if mask:
                data = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
            if op == OP_PING:
                self._write(OP_PONG, data)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                self.close(1000)
                raise Closed()
            if op in (OP_TEXT, OP_BIN):
                kind = "text" if op == OP_TEXT else "binary"
                msg = data
            elif op == OP_CONT:
                msg += data
                if len(msg) > self.max:
                    self.close(1009)
                    raise Closed()
            if fin:
                return kind, (msg.decode() if kind == "text" else msg)
