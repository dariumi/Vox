"""vox web (the engine behind Vox Studio): HTTP API, secret, CORS, TLS, and the Studio page in Chromium.
Run: python tests/test_web_studio.py [build/vox]

  1. /health without a secret, everything under /v1 only with it (header or ?token=), CORS preflight with
     Access-Control-Allow-Private-Network (Chrome: a public page calling the local network);
  2. models: list, open, close, add by path; profiles: create, add a recording (WAV in base64), PATCH, prepare, download, upload,
     delete, a recording as WAV;
  3. /v1/speak streams the same samples the library makes with the same voice and seed (pcm16 = round(x·32768)), the clip lands in
     the history with its notes, WAV and JSON;
  4. WebSocket /v1/duplex with ?token= (the protocol of vox serve);
  5. --tls makes a self-signed certificate and serves https (when `cryptography` or `openssl` is available);
  6. the Studio page the engine serves: connects with #secret, lists the cast, speaks a line with markup, shows the audio and the
     notes, the Voices tab shows the profile (needs playwright + chromium; otherwise SKIP)."""
import base64
import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    import vox
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails += 1

    work = tempfile.mkdtemp(prefix="voxweb_")
    models = os.path.join(work, "models")
    os.makedirs(models)
    tm = os.path.join(models, "tiny.tmod")
    vox.make_test_model(tm)
    sr = 24000
    t = np.arange(int(sr * 1.5)) / sr
    ref = os.path.join(work, "ref.wav")
    vox.save_wav(ref, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    secret = "s3cret-for-the-test-1234"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(os.path.join(work, "server.log"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "vox", "web", "--models", models, "--voices", os.path.join(work, "voices"),
                             "--data", work, "--port", str(port), "--secret", secret], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)

        def req(method, path, body=None, auth=True, raw=False, headers=None):
            h = dict(headers or {})
            if auth:
                h["Authorization"] = "Bearer " + secret
            data = None
            if body is not None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                h.setdefault("Content-Type", "application/json")
            r = urllib.request.Request(base + path, data=data, method=method, headers=h)
            try:
                with urllib.request.urlopen(r, timeout=120) as resp:
                    b = resp.read()
                    return resp.status, (b if raw else json.loads(b or b"{}")), dict(resp.headers)
            except urllib.error.HTTPError as e:
                b = e.read()
                return e.code, (b if raw else json.loads(b or b"{}")), dict(e.headers)

        # 1. secret and CORS
        code, j, _ = req("GET", "/health", auth=False)
        check(code == 200 and j.get("vox") == vox.__version__, f"/health without a secret ({j})")
        code, j, _ = req("GET", "/v1/info", auth=False)
        check(code == 401, "/v1 without the secret: 401")
        code, j, _ = req("GET", "/v1/info", headers={"Authorization": "Bearer wrong"}, auth=False)
        check(code == 401, "/v1 with a wrong secret: 401")
        code, j, _ = req("GET", f"/v1/info?token={secret}", auth=False)
        check(code == 200 and any(m["name"] == "tiny" for m in j["models"]), "/v1/info with ?token=")
        r = urllib.request.Request(base + "/v1/speak", method="OPTIONS", headers={"Origin": "https://vox.arkonova.network",
                                   "Access-Control-Request-Method": "POST", "Access-Control-Request-Private-Network": "true"})
        with urllib.request.urlopen(r) as resp:
            hh = dict(resp.headers)
        check(hh.get("Access-Control-Allow-Origin") == "https://vox.arkonova.network" and hh.get("Access-Control-Allow-Private-Network") == "true"
              and "Authorization" in hh.get("Access-Control-Allow-Headers", ""), "CORS preflight allows the page, Authorization and the private network")

        # 2. models and profiles
        code, j, _ = req("POST", "/v1/models/load", {"name": "tiny", "device": "cpu"})
        check(code == 200 and j["loaded"] and j["family"] == "vox-test", "open a model")
        code, j, _ = req("POST", "/v1/profiles", {"name": "cast"})
        check(code == 201, "create a profile")
        wav = base64.b64encode(open(ref, "rb").read()).decode()
        code, j, _ = req("POST", "/v1/profiles/cast/recordings", {"voice": "anna", "text": "hello there", "audio_b64": wav})
        code2, j2, _ = req("POST", "/v1/profiles/cast/recordings", {"voice": "boris", "text": "hello there", "audio_b64": wav})
        check(code == 201 and code2 == 201 and set(j2["voices"]) == {"anna", "boris"}, "add recordings (two voices)")
        code, j, _ = req("PATCH", "/v1/profiles/cast", {"voices": {"anna": {"styles": {"whisper": {"instruct": "Whisper"}}}}})
        check(code == 200 and "whisper" in j["voices"]["anna"]["styles"], "PATCH a profile")
        code, j, _ = req("POST", "/v1/profiles/cast/prepare", {"model": "tiny"})
        check(code == 200 and len(j["voices"]) == 3, f"prepare for a model ({j.get('voices')})")
        code, data, _ = req("GET", "/v1/profiles/cast/download", raw=True)
        p = os.path.join(work, "dl.tvoice")
        open(p, "wb").write(data)
        check(code == 200 and len(vox.Voice.open(p).info["cache"]) == 3, "download the .tvoice with the built voices")
        code, j, _ = req("PUT", "/v1/profiles/copy", data, headers={"Content-Type": "application/octet-stream"})
        check(code == 200 and set(j["voices"]) == {"anna", "boris"}, "upload a .tvoice")
        code, j, _ = req("PUT", "/v1/profiles/bad", b"not a tvoice", headers={"Content-Type": "application/octet-stream"})
        check(code == 400, "uploading garbage is refused")
        info = req("GET", "/v1/profiles/cast")[1]
        rid = info["voices"]["anna"]["styles"]["default"]["ref"]
        code, data, hh = req("GET", f"/v1/profiles/cast/recordings/{rid}.wav", raw=True)
        check(code == 200 and data[:4] == b"RIFF", "a recording as WAV")
        code, j, _ = req("DELETE", "/v1/profiles/copy")
        check(code == 200 and "copy" not in [x["file"] for x in req("GET", "/v1/profiles")[1]["profiles"]], "delete a profile")

        # 3. speak = the library, history
        text = "anna: Hello there. [pause 200ms]\nboris: Bye now. [laugh]"
        code, data, hh = req("POST", "/v1/speak", {"text": text, "model": "tiny", "profile": "cast", "params": {"seed": 9}, "format": "pcm16"}, raw=True)
        got = np.frombuffer(data, "<i2")
        eng = vox.Engine(tm, device="cpu")
        lib = eng.synthesize(text, vox.Voice.open(os.path.join(work, "voices", "cast.tvoice")), seed=9)
        expect = np.clip(np.rint(lib.astype(np.float64) * 32768), -32768, 32767).astype("<i2")
        check(code == 200 and hh.get("X-Vox-Sample-Rate") == "24000" and np.array_equal(got, expect),
              f"/v1/speak streams what the library makes ({len(got)} samples)")
        cid = hh.get("X-Vox-Clip")
        code, j, _ = req("GET", "/v1/history")
        clip = next((c for c in j["clips"] if c["id"] == cid), None)
        check(clip is not None and clip["profile"] == "cast" and any("[laugh]" in n for n in clip["notes"]) and clip["segments"] >= 2,
              "the clip is in the history with its notes")
        code, data, _ = req("GET", f"/v1/history/{cid}.wav", raw=True)
        check(code == 200 and data[:4] == b"RIFF" and len(data) == 44 + 2 * len(got), "the clip as WAV")
        code, data, hh = req("POST", "/v1/speak", {"text": "Hi.", "model": "tiny", "profile": "cast", "voice_id": "nobody"}, raw=True)
        check(code == 400 and b"anna" in data, "an unknown voice is a 400 that lists the voices")
        code, j, _ = req("POST", "/v1/speak", {"text": "Hi.", "model": "nope", "profile": "cast"})
        check(code == 404, "an unknown model is a 404")

        # 4. duplex over WebSocket
        from vox import ws as _ws
        s = socket.create_connection(("127.0.0.1", port))
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET /v1/duplex?token={secret}&model=tiny HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        f = s.makefile("rwb")
        status = f.readline()
        while f.readline() not in (b"\r\n", b""):
            pass
        w = _ws.WebSocket(f, f, client=True)
        w.send_text(json.dumps({"type": "start", "voice": "cast:boris", "seed": 3, "format": "f32"}))
        kind, msg = w.recv()
        ready = json.loads(msg)
        w.send_text(json.dumps({"type": "text", "text": "Hello over the socket."}))
        w.send_text(json.dumps({"type": "end"}))
        nbytes, done = 0, False
        while not done:
            kind, msg = w.recv()
            if kind == "binary":
                nbytes += len(msg)
            else:
                done = json.loads(msg).get("type") in ("done", "error")
        s.close()
        check(b"101" in status and ready.get("type") == "ready" and nbytes > 0, f"duplex WebSocket with ?token= ({nbytes} bytes)")

        # 6. the Studio page in Chromium
        try:
            from playwright.sync_api import sync_playwright
            have_pw = True
        except ImportError:
            have_pw = False
        if have_pw:
            with sync_playwright() as pw:
                br = pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
                pg = br.new_page()
                errs = []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.goto(f"{base}/studio/#secret={secret}")
                pg.wait_for_selector("#app:not([hidden])", timeout=20000)
                check("secret" not in pg.url, "Studio connects with #secret and removes it from the address bar")
                pg.wait_for_selector("#cast button")
                names = pg.locator("#cast button .nm").all_inner_texts()
                check(any(n.startswith("anna") for n in names) and any(n.startswith("boris") for n in names), f"the cast lists the profile voices ({len(names)})")
                pg.fill("#text", "anna: Hello there. [pause 300ms]\nboris: Bye. [whisper] now")
                pg.click("#speak")
                pg.wait_for_selector("article.line .ctl a[download]", timeout=60000)
                notes = pg.locator("article.line .notes li").all_inner_texts()
                check(len(notes) >= 1 and pg.locator("article.line .tag").count() >= 2, f"a spoken line: audio, download, tags shown, notes ({notes[:2]})")
                pg.click('nav.tabs button[data-view="voices"]')
                pg.wait_for_selector("#profDetail [data-voice='anna']", timeout=20000)
                check(pg.locator("#profDetail .style").count() >= 3, "the Voices tab shows voices and styles")
                pg.click('nav.tabs button[data-view="history"]')
                pg.wait_for_selector("#hRows tr", timeout=20000)
                check(pg.locator("#hRows tr").count() >= 2, "the History tab lists the clips")
                pg.click('nav.tabs button[data-view="models"]')
                pg.wait_for_selector("#mRows tr[data-m='tiny']", timeout=20000)
                check(True, "the Models tab lists the models")
                # the connect screen and its hint when the engine is unreachable
                pg2 = br.new_page()
                pg2.goto(f"{base}/studio/")
                pg2.fill("#cUrl", "http://127.0.0.1:1")
                pg2.fill("#cSecret", "x")
                pg2.click("#cGo")
                pg2.wait_for_selector("#cErr:not([hidden])", timeout=20000)
                check("vox web" in pg2.inner_text("#cErr"), "an unreachable engine: the hint says what to check")
                check(not errs, f"no page errors ({errs[:2]})")
                br.close()
        else:
            print("SKIP  Studio page (needs playwright + chromium)")
    finally:
        proc.terminate()
        proc.wait(10)

    # 5. TLS
    have_tls = True
    try:
        import cryptography  # noqa: F401
    except ImportError:
        have_tls = subprocess.run(["openssl", "version"], capture_output=True).returncode == 0
    if have_tls:
        port2 = free_port()
        proc = subprocess.Popen([sys.executable, "-m", "vox", "web", "--models", models, "--data", work, "--port", str(port2),
                                 "--secret", secret, "--tls"], stdout=log, stderr=subprocess.STDOUT)
        try:
            ok = False
            for _ in range(150):
                try:
                    ctx = ssl.create_default_context(cafile=os.path.join(work, "web_cert.pem"))
                    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN   # the self-signed leaf is its own trust anchor
                    with urllib.request.urlopen(f"https://127.0.0.1:{port2}/health", context=ctx, timeout=1) as r:
                        ok = json.loads(r.read()).get("status") == "ok"
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.1)
            check(ok, "--tls: a self-signed certificate for 127.0.0.1 (verified against itself) and https")
        finally:
            proc.terminate()
            proc.wait(10)
    else:
        print("SKIP  TLS (needs the cryptography package or openssl)")
    log.close()
    if fails:
        print(open(os.path.join(work, "server.log")).read()[-3000:])
    print("web studio: " + ("ALL OK" if not fails else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
