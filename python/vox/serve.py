"""vox serve: HTTP-сервер синтеза со стримингом (только стандартная библиотека + vox).

    python -m vox serve --model model.tmod --voices ./voices --port 8080

    GET  /health                    → {"status": "ok"}
    GET  /v1/info                   → информация о модели
    GET  /v1/voices                 → список голосов (*.tvoice из --voices)
    POST /v1/tts  {"text": "...", "voice": "имя", "language": "russian", "seed": 42, "temperature": 0.7,
                   "format": "wav" | "pcm16" | "f32", "decoder_mode": "stream" | "reference"}
                                                   → аудио, отдаётся чанками по мере генерации
    GET  /v1/duplex (WebSocket)     → двусторонний стриминг: текст на лету внутрь, звук на лету наружу (протокол ниже)
    POST /v1/voices {"name": "my_voice", "audio_b64": "<WAV в base64>", "ref_text": "текст референса",
                     "language": "russian", "xvector_only": false}
                                                   → создаёт голос нативно (без torch) и кладёт <name>.tvoice в --voices

/v1/duplex — WebSocket. Клиент → сервер, текстовые сообщения JSON:
    {"type":"start","voice":"имя","format":"pcm16"|"f32","commit":"auto"|"sentence"|"manual","idle_ms":600,"utterance_gap_ms":0,
     "on_conflict":"keep"|"raise"|"respeak","seed":42,...параметры генерации как у /v1/tts}   (первым)
    {"type":"text","text":"…"}   дописать текст          {"type":"flush"}   закрыть текущий кусок
    {"type":"set_text","text":"весь текст из поля ввода"} режим набора (правки ещё не озвученного хвоста)
    {"type":"end"}   текст закончился                       {"type":"cancel"}  оборвать
  Сервер → клиент: {"type":"ready","sample_rate":…,"format":…,"capabilities":{…}}; двоичные кадры — PCM (little-endian);
    {"type":"set_text_result",…}, {"type":"done","utterances":n,"stats":{…}}, {"type":"error","message":…}.

Для WAV заголовок пишется со «стриминговой» длиной (0xFFFFFFFF): плееры и ffmpeg читают такой поток.
curl -X POST localhost:8080/v1/tts -d '{"text":"Привет!","voice":"my_voice"}' | ffplay -i -
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import os
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import Engine, VoxError

_GEN_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty", "sub_temperature", "sub_top_k", "sub_top_p",
             "seed", "chunk_frames", "max_frames", "language", "decoder_mode", "decoder_context_frames")
_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def wav_stream_header(sr: int, bits: int, fmt: int) -> bytes:
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 0xFFFFFFFF, b"WAVE", b"fmt ", 16, fmt, 1, sr, sr * bits // 8,
                       bits // 8, bits, b"data", 0xFFFFFFFF)


class Server:
    def __init__(self, model, voices_dir, device="auto", threads=0):
        self.engine = Engine(model, device=device, threads=threads)
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self._voices = {}
        self._lock = threading.Lock()

    def list_voices(self):
        if not self.voices_dir or not self.voices_dir.is_dir():
            return []
        return sorted(p.stem for p in self.voices_dir.glob("*.tvoice"))

    def voice(self, name):
        if not self.voices_dir or not name or os.sep in name or name.startswith("."):
            raise KeyError(name)
        path = self.voices_dir / f"{name}.tvoice"
        if not path.is_file():
            raise KeyError(name)
        with self._lock:
            mtime = path.stat().st_mtime
            cached = self._voices.get(name)
            if not cached or cached[0] != mtime:
                self._voices[name] = (mtime, self.engine.load_voice(path))
            return self._voices[name][1]

    def duplex_session(self, sock):
        """Один WebSocket = одна двусторонняя сессия (см. протокол в шапке файла)."""
        from . import ws as _ws
        kind, raw = sock.recv()
        try:
            start = json.loads(raw) if kind == "text" else {}
            if start.get("type") != "start":
                raise ValueError('first message must be {"type":"start",...}')
            fmt = start.get("format", "pcm16")
            if fmt not in ("pcm16", "f32"):
                raise ValueError("format must be pcm16 or f32")
            voice = self.voice(start.get("voice", ""))
            params = {k: start[k] for k in _GEN_KEYS if k in start and k != "seed"}
            kw = {k: start[k] for k in ("commit", "idle_ms", "utterance_gap_ms", "on_conflict") if k in start}
            if "seed" in start:
                kw["seed"] = int(start["seed"])
            dx = self.engine.duplex(voice, **kw, **params)
        except KeyError as e:
            sock.send_text(json.dumps({"type": "error", "message": f"unknown voice: {e}"}))
            return
        except (ValueError, TypeError, VoxError, json.JSONDecodeError) as e:
            sock.send_text(json.dumps({"type": "error", "message": str(e)}))
            return
        sock.send_text(json.dumps({"type": "ready", "sample_rate": self.engine.sample_rate, "format": fmt,
                                   "capabilities": self.engine.capabilities}))

        def pump():   # звук наружу
            try:
                while True:
                    c = dx.read(timeout=0.05)
                    if c is None:
                        if dx.done:
                            break
                        continue
                    sock.send_binary(np.ascontiguousarray(c, dtype="<f4").tobytes() if fmt == "f32"
                                     else (np.clip(c, -1, 1) * 32767).astype("<i2").tobytes())
                sock.send_text(json.dumps({"type": "done", "utterances": dx.utterances, "stats": dx.stats}))
            except (_ws.Closed, OSError):
                pass
            except Exception as e:  # noqa: BLE001 — ошибка генерации
                try:
                    sock.send_text(json.dumps({"type": "error", "message": str(e)}))
                except Exception:  # noqa: BLE001
                    pass

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            while True:
                kind, raw = sock.recv()
                if kind != "text":
                    continue
                try:
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "text":
                        dx.write(str(msg.get("text", "")))
                    elif t == "flush":
                        dx.flush()
                    elif t == "set_text":
                        r = dx.set_text(str(msg.get("text", "")))
                        sock.send_text(json.dumps(dict(r, type="set_text_result"), ensure_ascii=False))
                    elif t == "end":
                        dx.end()
                    elif t == "cancel":
                        dx.cancel()
                        break
                    else:
                        sock.send_text(json.dumps({"type": "error", "message": f"unknown message type: {t}"}))
                except (VoxError, ValueError, TypeError, json.JSONDecodeError) as e:
                    sock.send_text(json.dumps({"type": "error", "message": str(e)}))
        except _ws.Closed:
            pass
        finally:
            dx.cancel() if not dx.done else None
            th.join(timeout=2)

    def create_voice(self, req: dict) -> str:
        from .audio import load_wav
        name = req.get("name", "")
        if not _NAME.match(name):
            raise ValueError("name must be 1-64 characters [A-Za-z0-9_-]")
        if not self.voices_dir:
            raise ValueError("server was started without --voices")
        raw = base64.b64decode(req["audio_b64"], validate=True)
        pcm, sr = load_wav(io.BytesIO(raw))
        xv = bool(req.get("xvector_only", False))
        text = req.get("ref_text")
        if not xv and not text:
            raise ValueError("ref_text is required unless xvector_only is true")
        v = self.engine.clone_voice((pcm, sr), None if xv else text, name=name, xvector_only=xv,
                                    **({"language": req["language"]} if req.get("language") else {}))
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.voices_dir / f".{name}.tvoice.tmp"
        v.save(tmp)
        os.replace(tmp, self.voices_dir / f"{name}.tvoice")
        return name


def make_handler(srv: Server):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("vox serve: " + fmt % args + "\n")

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _duplex(self):
            key = self.headers.get("Sec-WebSocket-Key")
            if "websocket" not in self.headers.get("Upgrade", "").lower() or not key:
                return self._json(400, {"error": "WebSocket upgrade required"})
            from . import ws as _ws
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", _ws.accept_key(key))
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            sock = _ws.WebSocket(self.rfile, self.wfile)
            try:
                srv.duplex_session(sock)
            except _ws.Closed:
                pass
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"vox serve: duplex failed: {e}\n")
            finally:
                sock.close()

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/duplex":
                return self._duplex()
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/info":
                self._json(200, srv.engine.info)
            elif self.path == "/v1/voices":
                self._json(200, {"voices": srv.list_voices()})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/v1/voices":
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    name = srv.create_voice(json.loads(self.rfile.read(n) or b"{}"))
                    return self._json(201, {"voice": name})
                except KeyError as e:
                    return self._json(400, {"error": f"missing: {e}"})
                except (ValueError, json.JSONDecodeError) as e:
                    return self._json(400, {"error": str(e)})
                except VoxError as e:
                    return self._json(500, {"error": str(e)})
            if self.path != "/v1/tts":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                text = req["text"]
                voice = srv.voice(req.get("voice", ""))
                fmt = req.get("format", "wav")
                if fmt not in ("wav", "pcm16", "f32"):
                    raise ValueError("format must be wav, pcm16 or f32")
                params = {k: req[k] for k in _GEN_KEYS if k in req}
            except KeyError as e:
                return self._json(404 if str(e).strip("'") != "text" else 400, {"error": f"missing or unknown: {e}"})
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            try:
                it = srv.engine.stream(text, voice, **params)
                first = next(it, None)  # ошибки генерации ловим до отправки заголовков
            except (VoxError, TypeError) as e:
                return self._json(400 if isinstance(e, TypeError) else 500, {"error": str(e)})
            sr = srv.engine.sample_rate
            self.send_response(200)
            self.send_header("Content-Type", {"wav": "audio/wav", "pcm16": "audio/L16", "f32": "application/octet-stream"}[fmt])
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Sample-Rate", str(sr))
            self.end_headers()

            def send(b: bytes):
                if b:
                    self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
                    self.wfile.flush()

            try:
                if fmt == "wav":
                    send(wav_stream_header(sr, 16, 1))
                for c in _chain([first] if first is not None else [], it):
                    if fmt == "f32":
                        send(np.ascontiguousarray(c, dtype="<f4").tobytes())
                    else:
                        send((np.clip(c, -1, 1) * 32767).astype("<i2").tobytes())
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # клиент ушёл; сессия освободится сборщиком (отмена в __exit__ не нужна — генератор закрыт)
            except VoxError as e:
                sys.stderr.write(f"vox serve: generation failed mid-stream: {e}\n")
                self.close_connection = True

    return H


def _chain(first, rest):
    for c in first:
        yield c
    for c in rest:
        yield c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", default="voices")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    srv = Server(a.model, a.voices, a.device, a.threads)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(srv))
    print(f"vox serve: http://{a.host}:{a.port}  model={a.model}  voices={a.voices} ({len(srv.list_voices())})", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
