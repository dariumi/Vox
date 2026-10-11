"""Мини-сервер, изображающий Hugging Face (API списка файлов + /resolve/), чтобы проверять загрузку моделей по ссылке без сети.
Корень — папка с репозиториями <org>/<name>/файлы...; репозитории из PRIVATE требуют заголовок Authorization: Bearer <token>."""
import http.server
import json
import os
import socketserver
import threading
from urllib.parse import unquote, urlparse


class MockHF:
    def __init__(self, root, private=(), token="hf_test", cdn=False):
        self.root, self.private, self.token, self.cdn = root, set(private), token, cdn   # cdn: /resolve/ отвечает редиректом на другой путь, как настоящий HF
        self.log = []                      # (path, authorization)
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="application/json", extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "authorization, range")
                self.send_header("Access-Control-Expose-Headers", "Content-Range, ETag, Accept-Ranges")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_OPTIONS(self):
                self._send(204)

            def _file(self, full):
                with open(full, "rb") as f:
                    data = f.read()
                etag = '"%d-%d"' % (len(data), int(os.path.getmtime(full)))
                rng = self.headers.get("Range")
                if rng:
                    outer.log.append(("RANGE " + self.path.split("?")[0], rng))
                if rng and rng.startswith("bytes="):
                    a, _, b = rng[6:].partition("-")
                    a = int(a)
                    b = min(int(b), len(data) - 1) if b else len(data) - 1
                    return self._send(206, data[a:b + 1], "application/octet-stream",
                                      {"Content-Range": f"bytes {a}-{b}/{len(data)}", "Accept-Ranges": "bytes", "ETag": etag})
                return self._send(200, data, "application/octet-stream", {"ETag": etag})

            def do_GET(self):
                u = urlparse(self.path)
                p = unquote(u.path)
                outer.log.append((p, self.headers.get("Authorization")))
                if p.startswith("/__cdn__/"):
                    full = os.path.normpath(os.path.join(outer.root, p[len("/__cdn__/"):]))
                    if not full.startswith(outer.root) or not os.path.isfile(full):
                        return self._send(404, b"not found")
                    return self._file(full)
                parts = p.strip("/").split("/")
                repo = None
                if p.startswith("/api/models/") and len(parts) >= 4:
                    repo = "/".join(parts[2:4])
                elif len(parts) >= 3:
                    repo = "/".join(parts[0:2])
                if repo is None or not os.path.isdir(os.path.join(outer.root, repo)):
                    return self._send(404, b'{"error":"Repository not found"}')
                if repo in outer.private and self.headers.get("Authorization") != "Bearer " + outer.token:
                    return self._send(401, b'{"error":"Invalid credentials"}')
                base = os.path.join(outer.root, repo)
                files = []
                for d, _, fs in os.walk(base):
                    for f in fs:
                        full = os.path.join(d, f)
                        files.append((os.path.relpath(full, base).replace(os.sep, "/"), os.path.getsize(full)))
                files.sort()
                if p.startswith("/api/models/"):
                    if len(parts) >= 5 and parts[4] == "tree":
                        return self._send(200, json.dumps([{"type": "file", "path": f, "size": s} for f, s in files]).encode())
                    if len(parts) >= 5 and parts[4] == "revision":
                        return self._send(200, json.dumps({"id": repo, "siblings": [{"rfilename": f} for f, _ in files]}).encode())
                    return self._send(200, json.dumps({"id": repo, "siblings": [{"rfilename": f} for f, _ in files]}).encode())
                if len(parts) >= 5 and parts[2] == "resolve":
                    rel = "/".join(parts[4:])
                    full = os.path.normpath(os.path.join(base, rel))
                    if not full.startswith(base) or not os.path.isfile(full):
                        return self._send(404, b"Entry not found")
                    if outer.cdn:
                        return self._send(302, b"", "text/plain", {"Location": outer.url + "/__cdn__/" + repo + "/" + rel + "?sig=1"})
                    return self._file(full)
                self._send(404, b"not found")

            do_HEAD = do_GET

        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def build_repos(work, tmp):
    """Раскладывает мок-репозитории HF из крошечной модели F5 (work) в tmp/hf; ссылочные .tmod из Python — в tmp/refs. → (root, refs)."""
    import shutil
    import sys
    import torch
    from safetensors.torch import load_file
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(os.path.dirname(here), "python")]
    from vox.build_f5 import build
    refs = os.path.join(tmp, "refs")
    os.makedirs(refs)
    ck, vc = os.path.join(work, "ckpt"), os.path.join(work, "vocos")
    vocos_w = vc if os.path.isfile(os.path.join(vc, "pytorch_model.bin")) else vc
    for name, kw in (("bf16", dict(dtype="bf16")), ("f16q8", dict(dtype="f16", quant="q8"))):
        build(os.path.join(ck, "model.safetensors"), os.path.join(ck, "vocab.txt"), vocos_w, os.path.join(refs, f"py_{name}.tmod"), dim_head=32,
              variant="F5TTS_v1_Base", sha=False, **kw)
    # репозитории: чекпойнт .safetensors, .pt, вокодер .bin (настоящий torch.save)
    repos = os.path.join(tmp, "hf")

    def put(rel, data=None, src=None):
        p = os.path.join(repos, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if src:
            shutil.copy(src, p)
        else:
            open(p, "wb").write(data)

    sd = load_file(os.path.join(ck, "model.safetensors"))
    put("mock/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors", src=os.path.join(ck, "model.safetensors"))
    put("mock/F5-TTS/F5TTS_v1_Base/vocab.txt", src=os.path.join(ck, "vocab.txt"))
    pt = os.path.join(tmp, "model.pt")
    torch.save({"ema_model_state_dict": {k: v.clone() for k, v in sd.items()}, "update": 1200000, "step": 1200000}, pt)
    put("mock/F5-TTS/F5TTS_Base/model_1200000.pt", src=pt)
    put("mock/F5-TTS/F5TTS_Base/vocab.txt", src=os.path.join(ck, "vocab.txt"))
    put("mock/F5-TTS/README.md", b"# mock")
    vsd = load_file(os.path.join(vc, "pytorch_model.safetensors")) if os.path.exists(os.path.join(vc, "pytorch_model.safetensors")) else None
    vbin = os.path.join(tmp, "pytorch_model.bin")
    if vsd is not None:
        torch.save({k: v.clone() for k, v in vsd.items()}, vbin)
    else:
        shutil.copy(os.path.join(vc, "pytorch_model.bin"), vbin)
    put("charactr/vocos-mel-24khz/pytorch_model.bin", src=vbin)
    put("charactr/vocos-mel-24khz/config.yaml", src=os.path.join(vc, "config.yaml"))   # padding головы ISTFT читается отсюда
    put("mock/vocos/pytorch_model.bin", src=vbin)
    put("mock/f5-tmod/model.q8.tmod", src=os.path.join(refs, "py_f16q8.tmod"))
    put("mock/f5-tmod/model.bf16.tmod", src=os.path.join(refs, "py_bf16.tmod"))
    put("mock/private/model.bf16.tmod", src=os.path.join(refs, "py_bf16.tmod"))
    put("mock/Qwen3-TTS-12Hz-0.6B-Base/config.json", b"{}")
    put("mock/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors", b"x")
    put("mock/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/model.safetensors", b"x")
    return repos, refs
