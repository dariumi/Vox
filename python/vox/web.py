"""vox web: the engine as a backend for Vox Studio — the page at https://vox.arkonova.network/studio/ (or the copy this server serves
itself) — and for any other client over HTTP.

    vox web --models ./models --voices ./voices                (or: python -m vox web …)

prints links with a secret. The page lives on the web, the engine runs on your machine, and the page talks to the engine with HTTP
requests — the way a Clash dashboard talks to its core. The secret travels in the link's #fragment, which browsers never send to the
site hosting the page. Every /v1 request needs it: "Authorization: Bearer <secret>" (or ?token=<secret> for <audio src> and WebSockets).

Reaching the engine from a page:
  * the same computer: an https page may call http://127.0.0.1 / http://localhost (Chrome may ask once to allow local network access);
  * another computer (LAN or internet): an https page may not call plain http on another host. Use `--tls` (a self-signed certificate is
    made for this machine's names and addresses; open the engine's https:// address once and accept it) or your own --cert/--key, or
    open the Studio the engine serves itself (http://HOST:PORT/studio/), or allow "insecure content" for the site in the browser.

    GET    /health                                   {"status","vox","studio"}            (no secret)
    GET    /studio/                                  the Studio page served by this engine (no secret: it asks for it)
    GET    /v1/info                                  version, devices, models, profiles
    GET    /v1/models;  POST /v1/models/load {"name","device"?};  POST /v1/models/unload {"name"};  POST /v1/models/add {"path","name"?}
    GET    /v1/models/<name>                         capabilities, speakers, languages, plan (loads nothing when not loaded)
    GET    /v1/profiles                              profile summaries;  POST /v1/profiles {"name"} creates an empty one
    GET    /v1/profiles/<p>                          the profile (voices, styles, recordings, cache);  DELETE removes the file
    PATCH  /v1/profiles/<p>   {JSON merge patch}     voices / styles / speakers / descriptions / options (see vox.h)
    PUT    /v1/profiles/<p>   (raw .tvoice bytes)    upload;   GET /v1/profiles/<p>/download → the .tvoice
    POST   /v1/profiles/<p>/recordings {"voice","style"?,"text"?,"language"?,"audio_b64" (WAV)}
    GET    /v1/profiles/<p>/recordings/<id>.wav
    POST   /v1/profiles/<p>/prepare {"model","voice"?,"style"?}   build the voices for a model now (kept in the file)
    POST   /v1/speak  {"text","model", "profile"? + "voice_id"?/"style"?  | "speaker"?/"instruct"?, "markup"?, "language"?,
                       "params"?: {seed, temperature, speed, nfe_step, …}, "format": "pcm16" (default) | "f32" | "wav"}
                      → audio streamed as it is synthesized (X-Vox-Sample-Rate, X-Vox-Clip); the clip goes to the history
    GET    /v1/history;  GET /v1/history/<id>.wav;  GET /v1/history/<id>;  DELETE /v1/history/<id>
    GET    /v1/duplex?token=…&model=…               WebSocket, the protocol of `vox serve` (start/text/set_text/flush/end/cancel)
"""
from __future__ import annotations

import argparse
import base64
import hmac
import io
import ipaddress
import json
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import numpy as np

from . import Engine, Voice, VoxError, __version__
from .audio import load_wav

STUDIO_URL = "https://vox.arkonova.network/studio/"
MAX_BODY = 128 << 20
_NAME = re.compile(r"^[\w][\w .\-]{0,63}$", re.UNICODE)
_PARAM_KEYS = ("seed", "temperature", "top_k", "top_p", "repetition_penalty", "sub_temperature", "sub_top_k", "sub_top_p",
               "max_frames", "chunk_frames", "language", "speed", "nfe_step", "cfg_strength", "sway_sampling_coef", "target_rms",
               "cross_fade_duration", "decoder_mode", "decoder_context_frames", "phonemes", "lang")


def home_dir() -> Path:
    d = Path(os.environ.get("VOX_HOME") or (Path.home() / ".vox"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_or_make_secret(path: Path, renew: bool = False) -> str:
    if not renew and path.exists():
        s = path.read_text().strip()
        if len(s) >= 16:
            return s
    s = secrets.token_urlsafe(24)
    path.write_text(s + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return s


class HttpError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    import struct
    payload = np.clip(np.rint(np.asarray(pcm, np.float64) * 32768.0), -32768, 32767).astype("<i2").tobytes()
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(payload), b"WAVE", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
                       b"data", len(payload)) + payload


class Studio:
    """State behind the API: models (opened on demand), profiles on disk (one handle per file, shared by all models so builds
    accumulate in it), the clip history."""

    def __init__(self, models: dict, voices_dir: Path, data_dir: Path, device: str = "auto", threads: int = 0):
        self.models = dict(models)              # name → path
        self.engines: dict[str, Engine] = {}
        self.devices: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.device, self.threads = device, threads
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = data_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._profiles: dict[str, tuple] = {}   # name → (mtime, Voice)
        self._speakers: dict[tuple, Voice] = {}

    # ---------------- models
    def model_rows(self) -> list:
        from . import inspect
        out = []
        for name, path in sorted(self.models.items()):
            row = {"name": name, "path": str(path), "loaded": name in self.engines}
            try:
                row["size_mb"] = round(os.path.getsize(path) / 2**20, 1)
                m = inspect(path)
                row.update(family=m.get("family"), variant=m.get("variant"), capabilities=m.get("capabilities", {}))
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            if name in self.engines:
                info = self.engines[name].info
                row.update(device=info.get("device"), backend=info.get("backend"), capabilities=info.get("capabilities", {}),
                           sample_rate=info.get("sample_rate"))
            if name in self.errors:
                row["error"] = self.errors[name]
            out.append(row)
        return out

    def engine(self, name: str | None) -> Engine:
        if not name:
            if len(self.models) == 1:
                name = next(iter(self.models))
            else:
                raise HttpError(400, "say which model: \"model\": one of " + ", ".join(sorted(self.models)))
        if name not in self.models:
            raise HttpError(404, f"unknown model {name!r}; known: {', '.join(sorted(self.models))}")
        with self.lock:
            e = self.engines.get(name)
            if e is None:
                try:
                    e = Engine(self.models[name], device=self.devices.get(name, self.device), threads=self.threads)
                except VoxError as ex:
                    self.errors[name] = str(ex)
                    raise HttpError(500, f"cannot open {name}: {ex}")
                self.errors.pop(name, None)
                self.engines[name] = e
            return e

    def load_model(self, name: str, device: str | None) -> dict:
        with self.lock:
            if device and self.devices.get(name) != device:
                self.devices[name] = device
                old = self.engines.pop(name, None)
                if old:
                    old.close()
        self.engine(name)
        return next(r for r in self.model_rows() if r["name"] == name)

    def unload_model(self, name: str) -> None:
        with self.lock:
            e = self.engines.pop(name, None)
            if e:
                e.close()
            self._speakers = {k: v for k, v in self._speakers.items() if k[0] != name}

    def add_model(self, path: str, name: str | None) -> dict:
        p = Path(path).expanduser()
        if not p.is_file():
            raise HttpError(400, f"no such file on the engine's machine: {path}")
        name = name or p.stem
        if not _NAME.match(name):
            raise HttpError(400, "bad model name")
        with self.lock:
            self.models[name] = str(p)
        return next(r for r in self.model_rows() if r["name"] == name)

    # ---------------- profiles
    def _ppath(self, name: str) -> Path:
        if not _NAME.match(name or ""):
            raise HttpError(400, "bad profile name (letters, digits, space, . _ -; up to 64)")
        return self.voices_dir / f"{name}.tvoice"

    def profile_names(self) -> list:
        return sorted(p.stem for p in self.voices_dir.glob("*.tvoice"))

    def profile(self, name: str) -> Voice:
        path = self._ppath(name)
        if not path.exists():
            raise HttpError(404, f"no profile {name!r}")
        with self.lock:
            mt = path.stat().st_mtime_ns
            cached = self._profiles.get(name)
            if cached and cached[0] == mt:
                return cached[1]
            v = Voice.open(path)
            self._profiles[name] = (mt, v)
            return v

    def save_profile(self, name: str, v: Voice) -> None:
        path = self._ppath(name)
        with self.lock:
            tmp = path.with_name(f".{path.name}.tmp")
            v.save(tmp)
            os.replace(tmp, path)
            self._profiles[name] = (path.stat().st_mtime_ns, v)

    def profile_summary(self, name: str) -> dict:
        i = self.profile(name).info
        return {"file": name, "name": i.get("name"), "default_voice": i.get("default_voice"),
                "voices": {k: {"styles": list(v.get("styles", {})), "speakers": v.get("speakers", {}), "description": v.get("description", ""),
                               "language": v.get("language", "auto")} for k, v in i.get("voices", {}).items()},
                "built_for": sorted({f"{c['model']['family']}/{c['model']['variant']}" for c in i.get("cache", [])}),
                "recordings": len(i.get("recordings", {}))}

    def recording_wav(self, name: str, rid: str) -> bytes:
        # recordings are not exposed through the C ABI one by one: read them from the file (safetensors), it is small
        import struct
        path = self._ppath(name)
        info = self.profile(name).info
        if rid not in info.get("recordings", {}):
            raise HttpError(404, f"no recording {rid!r}")
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            meta = json.loads(hdr.get("__metadata__", {}).get("profile", "{}"))
            tname = meta.get("recordings", {}).get(rid, {}).get("tensor", f"ref.{rid}")
            if tname not in hdr:
                raise HttpError(404, "the recording is not saved in the file yet")
            a, b = hdr[tname]["data_offsets"]
            f.seek(8 + n + a)
            return f.read(b - a)

    # ---------------- voices for a synthesis
    def voice_for(self, req: dict, eng: Engine, model: str):
        if req.get("profile"):
            return self.profile(req["profile"]), req["profile"]
        spk, ins = req.get("speaker") or None, req.get("instruct") or None
        if spk or ins:
            key = (model, spk, ins)
            with self.lock:
                if key not in self._speakers:
                    try:
                        self._speakers[key] = eng.speaker_voice(spk, ins)
                    except VoxError as e:
                        raise HttpError(400, str(e))
                return self._speakers[key], None
        raise HttpError(400, "give \"profile\" (and optionally \"voice_id\" / \"style\"), or \"speaker\" / \"instruct\"")

    # ---------------- history
    def history(self) -> list:
        rows = []
        for p in sorted(self.history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:500]:
            try:
                rows.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        return rows

    def clip_path(self, cid: str, ext: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", cid or ""):
            raise HttpError(400, "bad clip id")
        p = self.history_dir / f"{cid}.{ext}"
        if not p.exists():
            raise HttpError(404, "no such clip")
        return p


# ---------------- TLS: a self-signed certificate for this machine
def local_names() -> tuple:
    names = {"localhost", socket.gethostname()}
    ips = {"127.0.0.1", "::1"}
    try:
        for fam, _, _, _, addr in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(addr[0])
    except OSError:
        pass
    try:   # the address of the default route (the LAN address)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(n for n in names if n), sorted(ips)


def make_self_signed(cert: Path, key: Path) -> str:
    """Writes a self-signed certificate valid for localhost, the host name and this machine's addresses. Tries the `cryptography`
    package, then the `openssl` command. Returns how it was made."""
    names, ips = local_names()
    try:
        import datetime
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        k = ec.generate_private_key(ec.SECP256R1())
        subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vox web (self-signed)")])
        san = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
        now = datetime.datetime.now(datetime.timezone.utc)
        c = (x509.CertificateBuilder().subject_name(subj).issuer_name(subj).public_key(k.public_key())
             .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
             .not_valid_after(now + datetime.timedelta(days=825)).add_extension(x509.SubjectAlternativeName(san), critical=False)
             .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True).sign(k, hashes.SHA256()))
        key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
        how = "cryptography"
    except ImportError:
        san = ",".join([f"DNS:{n}" for n in names] + [f"IP:{i}" for i in ips])
        r = subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
                            "-keyout", str(key), "-out", str(cert), "-days", "825", "-subj", "/CN=vox web (self-signed)",
                            "-addext", f"subjectAltName={san}"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("--tls needs the `cryptography` Python package or the `openssl` command to make a certificate "
                             "(or pass your own --cert/--key):\n" + r.stderr[-800:])
        how = "openssl"
    try:
        os.chmod(key, 0o600)
    except OSError:
        pass
    return how


# ---------------- HTTP
def studio_html() -> bytes:
    p = Path(__file__).with_name("studio.html")
    return p.read_bytes() if p.exists() else b"<!doctype html><title>Vox Studio</title><p>studio.html is missing from this vox package."


def make_handler(st: Studio, secret: str):
    from .serve import Server as _Serve

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"vox/{__version__}"

        def log_message(self, fmt, *args):
            if os.environ.get("VOX_WEB_LOG"):
                sys.stderr.write("vox web: " + fmt % args + "\n")

        # ---- plumbing
        def _cors(self):
            origin = self.headers.get("Origin")
            self.send_header("Access-Control-Allow-Origin", origin or "*")
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
            self.send_header("Access-Control-Expose-Headers", "X-Vox-Sample-Rate, X-Vox-Clip, X-Vox-Format")
            self.send_header("Access-Control-Allow-Private-Network", "true")   # Chrome: public page → local network

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise HttpError(413, "request too large")
            return self.rfile.read(n) if n else b""

        def _jbody(self) -> dict:
            raw = self._body()
            try:
                j = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                raise HttpError(400, f"bad JSON: {e}")
            if not isinstance(j, dict):
                raise HttpError(400, "JSON object expected")
            return j

        def _authorized(self, q: dict) -> bool:
            tok = ""
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                tok = auth[7:].strip()
            tok = tok or (q.get("token") or [""])[0]
            return bool(tok) and hmac.compare_digest(tok.encode(), secret.encode())

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _route(self, method: str):
            u = urlsplit(self.path)
            path, q = unquote(u.path), parse_qs(u.query)
            try:
                if method == "GET" and path == "/health":
                    return self._json(200, {"status": "ok", "vox": __version__, "studio": 1})
                if method == "GET" and path in ("/", "/studio", "/studio/", "/studio/index.html"):
                    if path in ("/", "/studio"):
                        self.send_response(302)
                        self.send_header("Location", "/studio/")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    return self._send(200, studio_html(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
                if not path.startswith("/v1/"):
                    raise HttpError(404, "not found")
                if not self._authorized(q):
                    raise HttpError(401, "wrong or missing secret (Authorization: Bearer <secret>)")
                return self._api(method, path[4:], q)
            except HttpError as e:
                return self._json(e.code, {"error": str(e)})
            except VoxError as e:
                return self._json(400 if e.status in (-6, -8) else 500, {"error": e.message, "status": e.status})
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._route("GET")

        def do_HEAD(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_PUT(self):
            self._route("PUT")

        def do_PATCH(self):
            self._route("PATCH")

        def do_DELETE(self):
            self._route("DELETE")

        # ---- API
        def _api(self, m: str, p: str, q: dict):
            parts = [x for x in p.split("/") if x]
            if m == "GET" and parts == ["info"]:
                from . import devices
                return self._json(200, {"vox": __version__, "devices": devices(), "models": st.model_rows(),
                                        "profiles": [st.profile_summary(n) for n in st.profile_names()]})
            if parts[:1] == ["models"]:
                return self._models(m, parts[1:])
            if parts[:1] == ["profiles"]:
                return self._profiles(m, parts[1:])
            if m == "POST" and parts == ["speak"]:
                return self._speak(self._jbody())
            if parts[:1] == ["history"]:
                return self._history(m, parts[1:])
            if m == "GET" and parts == ["duplex"]:
                return self._duplex(q)
            raise HttpError(404, f"no such endpoint: {m} /v1/{p}")

        def _models(self, m, parts):
            if m == "GET" and not parts:
                return self._json(200, {"models": st.model_rows()})
            if m == "POST" and parts == ["load"]:
                j = self._jbody()
                return self._json(200, st.load_model(j.get("name", ""), j.get("device")))
            if m == "POST" and parts == ["unload"]:
                st.unload_model(self._jbody().get("name", ""))
                return self._json(200, {"ok": True})
            if m == "POST" and parts == ["add"]:
                j = self._jbody()
                return self._json(200, st.add_model(j.get("path", ""), j.get("name")))
            if m == "GET" and len(parts) == 1:
                row = next((r for r in st.model_rows() if r["name"] == parts[0]), None)
                if not row:
                    raise HttpError(404, f"unknown model {parts[0]!r}")
                if row["loaded"]:
                    row["plan"] = st.engines[parts[0]].plan
                return self._json(200, row)
            raise HttpError(404, "no such models endpoint")

        def _profiles(self, m, parts):
            if not parts:
                if m == "GET":
                    return self._json(200, {"profiles": [st.profile_summary(n) for n in st.profile_names()]})
                if m == "POST":
                    name = self._jbody().get("name", "")
                    path = st._ppath(name)
                    if path.exists():
                        raise HttpError(409, f"profile {name!r} exists")
                    from . import Profile
                    st.save_profile(name, Profile(name))
                    return self._json(201, st.profile_summary(name))
                raise HttpError(405, "GET or POST")
            name = parts[0]
            if len(parts) == 1:
                if m == "GET":
                    return self._json(200, dict(st.profile(name).info, file=name))
                if m == "DELETE":
                    path = st._ppath(name)
                    if not path.exists():
                        raise HttpError(404, f"no profile {name!r}")
                    path.unlink()
                    st._profiles.pop(name, None)
                    return self._json(200, {"ok": True})
                if m == "PATCH":
                    v = st.profile(name)
                    v.update(self._jbody())
                    st.save_profile(name, v)
                    return self._json(200, dict(v.info, file=name))
                if m == "PUT":
                    raw = self._body()
                    path = st._ppath(name)
                    tmp = path.with_name(f".{path.name}.up")
                    tmp.write_bytes(raw)
                    try:
                        Voice.open(tmp)
                    except VoxError as e:
                        tmp.unlink()
                        raise HttpError(400, f"not a .tvoice: {e.message}")
                    os.replace(tmp, path)
                    st._profiles.pop(name, None)
                    return self._json(200, st.profile_summary(name))
            if m == "GET" and parts[1:] == ["download"]:
                st.profile(name)
                data = st._ppath(name).read_bytes()
                return self._send(200, data, "application/octet-stream",
                                  {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.tvoice"})
            if m == "POST" and parts[1:] == ["recordings"]:
                j = self._jbody()
                try:
                    raw = base64.b64decode(j.get("audio_b64", ""), validate=True)
                    pcm, sr = load_wav(io.BytesIO(raw))
                except Exception as e:  # noqa: BLE001
                    raise HttpError(400, f"audio_b64 must be a WAV file in base64: {e}")
                if not j.get("voice"):
                    raise HttpError(400, "\"voice\" is required")
                v = st.profile(name)
                v.add_recording(j["voice"], (pcm, sr), j.get("text") or None, style=j.get("style") or None, language=j.get("language") or None)
                st.save_profile(name, v)
                return self._json(201, dict(v.info, file=name))
            if m == "GET" and len(parts) == 3 and parts[1] == "recordings":
                rid = parts[2][:-4] if parts[2].endswith(".wav") else parts[2]
                return self._send(200, st.recording_wav(name, rid), "audio/wav")
            if m == "POST" and parts[1:] == ["prepare"]:
                j = self._jbody()
                eng = st.engine(j.get("model"))
                v = st.profile(name)
                r = v.prepare(eng, j.get("voice") or None, j.get("style") or None)
                st.save_profile(name, v)
                return self._json(200, r)
            raise HttpError(404, "no such profiles endpoint")

        def _speak(self, j: dict):
            text = j.get("text")
            if not isinstance(text, str) or not text.strip():
                raise HttpError(400, "\"text\" is required")
            model = j.get("model") or (next(iter(st.models)) if len(st.models) == 1 else None)
            eng = st.engine(model)
            voice, prof = st.voice_for(j, eng, model)
            fmt = j.get("format", "pcm16")
            if fmt not in ("pcm16", "f32", "wav"):
                raise HttpError(400, "format must be pcm16, f32 or wav")
            params = {k: v for k, v in (j.get("params") or {}).items() if k in _PARAM_KEYS and v is not None}
            if "seed" not in params:
                params["seed"] = int.from_bytes(os.urandom(4), "little") or 1
            for k in ("voice_id", "style", "markup", "language"):
                if j.get(k):
                    params[k] = j[k]
            n_cache = len(voice.info.get("cache", [])) if prof else 0
            t0 = time.perf_counter()
            s = eng.session(voice, **params)
            s.feed(text)
            it = iter(s)
            first = next(it, None)   # errors before the headers go out
            sr = eng.sample_rate
            cid = uuid.uuid4().hex
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", {"wav": "audio/wav", "pcm16": "audio/L16", "f32": "application/octet-stream"}[fmt])
            self.send_header("X-Vox-Sample-Rate", str(sr))
            self.send_header("X-Vox-Clip", cid)
            self.send_header("X-Vox-Format", fmt)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def chunk(b: bytes):
                if b:
                    self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
                    self.wfile.flush()

            if fmt == "wav":   # a streaming WAV: the header claims the maximum length
                import struct
                chunk(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 0xFFFFFFFF, b"WAVE", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16, b"data", 0xFFFFFFF0))
            parts, first_t = [], None
            client_gone = False
            pending = [first] if first is not None else []
            while True:
                c = pending.pop() if pending else next(it, None)
                if c is None:
                    break
                if first_t is None:
                    first_t = time.perf_counter() - t0
                parts.append(c)
                if client_gone:
                    continue
                try:
                    chunk(np.ascontiguousarray(c, "<f4").tobytes() if fmt == "f32" else
                          np.clip(np.rint(np.asarray(c, np.float64) * 32768.0), -32768, 32767).astype("<i2").tobytes())
                except (BrokenPipeError, ConnectionResetError):
                    client_gone = True
                    s.cancel()
                    break
            if not client_gone:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    client_gone = True
            pcm = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            total = time.perf_counter() - t0
            info = s.info
            meta = {"id": cid, "text": text, "model": model, "profile": prof, "voice_id": j.get("voice_id"), "style": j.get("style"),
                    "speaker": j.get("speaker"), "instruct": j.get("instruct"), "params": params, "created": time.time(),
                    "seconds": len(pcm) / sr, "compute_s": total, "first_chunk_s": first_t, "rtf": total / (len(pcm) / sr) if len(pcm) else None,
                    "notes": info.get("notes", []), "segments": info.get("segment", 0) + 1, "complete": not client_gone}
            (st.history_dir / f"{cid}.wav").write_bytes(_wav_bytes(pcm, sr))
            (st.history_dir / f"{cid}.json").write_text(json.dumps(meta, ensure_ascii=False))
            if prof and len(voice.info.get("cache", [])) != n_cache:   # voices built for this model: keep them in the file
                try:
                    st.save_profile(prof, voice)
                except (OSError, VoxError):
                    pass
            self.close_connection = client_gone

        def _history(self, m, parts):
            if m == "GET" and not parts:
                return self._json(200, {"clips": st.history()})
            if len(parts) == 1:
                cid = parts[0]
                if m == "GET" and cid.endswith(".wav"):
                    return self._send(200, st.clip_path(cid[:-4], "wav").read_bytes(), "audio/wav")
                if m == "GET":
                    return self._json(200, json.loads(st.clip_path(cid, "json").read_text()))
                if m == "DELETE":
                    for ext in ("wav", "json"):
                        try:
                            st.clip_path(cid, ext).unlink()
                        except HttpError:
                            pass
                    return self._json(200, {"ok": True})
            raise HttpError(404, "no such history endpoint")

        def _duplex(self, q: dict):
            key = self.headers.get("Sec-WebSocket-Key")
            if "websocket" not in self.headers.get("Upgrade", "").lower() or not key:
                raise HttpError(400, "WebSocket upgrade required")
            model = (q.get("model") or [None])[0]
            eng = st.engine(model)
            from . import ws as _ws

            class Adapter:   # what serve.Server.duplex_session needs: .engine and .voice_params(name)
                engine = eng

                @staticmethod
                def voice_params(name):
                    if ":" in (name or ""):
                        f, vid = name.split(":", 1)
                        return st.profile(f), {"voice_id": vid}
                    if name and st._ppath(name).exists():
                        return st.profile(name), {}
                    return eng.speaker_voice(name), {}

            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", _ws.accept_key(key))
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            sock = _ws.WebSocket(self.rfile, self.wfile)
            try:
                _Serve.duplex_session(Adapter, sock)
            except _ws.Closed:
                pass
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"vox web: duplex failed: {e}\n")
            finally:
                sock.close()

    return H


def find_models(specs: list) -> dict:
    out = {}
    for spec in specs:
        name = None
        if "=" in spec and not os.path.exists(spec):
            name, spec = spec.split("=", 1)
        p = Path(spec).expanduser()
        files = sorted(p.glob("*.tmod")) if p.is_dir() else [p]
        for f in files:
            if not f.is_file():
                raise SystemExit(f"vox web: no such model: {f}")
            out[name if name and len(files) == 1 else f.stem] = str(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vox web", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], help="model files or folders with .tmod (name=path to rename)")
    ap.add_argument("--voices", default="", help="folder with profiles (.tvoice); default ~/.vox/voices")
    ap.add_argument("--data", default="", help="folder for the history and the secret; default ~/.vox")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 = every interface: the LAN can reach it)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--secret", default="", help="use this secret (default: a stored random one, see --new-secret)")
    ap.add_argument("--new-secret", action="store_true", help="make a new secret (old links stop working)")
    ap.add_argument("--tls", action="store_true", help="HTTPS with a self-signed certificate for this machine (made once, kept in --data)")
    ap.add_argument("--cert", default="", help="HTTPS with your certificate (PEM)")
    ap.add_argument("--key", default="", help="the key of --cert (PEM)")
    ap.add_argument("--studio-url", default=STUDIO_URL, help="the Studio page the printed link opens")
    a = ap.parse_args(argv)
    data = Path(a.data).expanduser() if a.data else home_dir()
    data.mkdir(parents=True, exist_ok=True)
    voices = Path(a.voices).expanduser() if a.voices else data / "voices"
    secret = a.secret or load_or_make_secret(data / "web_secret", a.new_secret)
    models = find_models(a.models)
    if not models:
        print("vox web: no models yet (--models DIR|FILE …); they can be added from Studio by path", file=sys.stderr)
    st = Studio(models, voices, data, a.device, a.threads)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(st, secret))
    srv.daemon_threads = True
    scheme = "http"
    if a.tls or a.cert:
        cert, key = (Path(a.cert), Path(a.key or a.cert)) if a.cert else (data / "web_cert.pem", data / "web_key.pem")
        if not a.cert and not cert.exists():
            how = make_self_signed(cert, key)
            print(f"vox web: made a self-signed certificate ({how}): {cert}", file=sys.stderr)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert), str(key))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    hosts = ["127.0.0.1"] if a.host in ("127.0.0.1", "localhost") else (local_names()[1] if a.host in ("0.0.0.0", "::") else [a.host])
    hosts = [h for h in hosts if ":" not in h] or ["127.0.0.1"]
    print(f"vox web {__version__}: {len(models)} model(s), profiles in {voices}")
    for h in hosts:
        base = f"{scheme}://{h}:{a.port}"
        print(f"\n  Vox Studio:  {a.studio_url}#url={quote(base, safe='')}&secret={secret}")
        print(f"  this engine: {base}/studio/#secret={secret}")
    print(f"\n  API: {scheme}://{hosts[0]}:{a.port}/v1/…  with  Authorization: Bearer {secret}")
    if scheme == "https" and not a.cert:
        print(f"  self-signed: open {scheme}://{hosts[0]}:{a.port}/health once in the browser and accept the certificate")
    if a.host not in ("127.0.0.1", "localhost") and scheme == "http":
        print("  note: from another computer an https page cannot reach plain http; use --tls, or the Studio this engine serves")
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
