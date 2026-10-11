"""Двусторонний стриминг на тестовой модели vox-test (Python-слой Duplex + политики коммита движка).
Запуск: python tests/test_duplex.py [build/vox]"""
import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

import vox

HERE = os.path.dirname(os.path.abspath(__file__))
fails = 0


def check(ok, what):
    global fails
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        fails += 1


def ref_audio(sr=24000, secs=3):
    t = np.arange(sr * secs, dtype=np.float32)
    return (0.3 * np.sin(t * 0.05) * (t % 4000) / 4000).astype(np.float32), sr


def ws_check(eng, voice, d, text, want):
    """WebSocket /v1/duplex сервера: клиент на сыром сокете (рукопожатие и кадры руками)."""
    import base64
    import json
    import socket
    import struct
    import urllib.request
    from vox import ws
    from vox.serve import Server, make_handler
    from http.server import ThreadingHTTPServer
    vdir = os.path.join(d, "voices")
    os.makedirs(vdir)
    voice.save(os.path.join(vdir, "demo.tvoice"))
    srv = Server(os.path.join(d, "test.tmod"), vdir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(srv))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def connect():
        sock = socket.create_connection(("127.0.0.1", port), timeout=20)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f"GET /v1/duplex HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        f = sock.makefile("rwb", buffering=0)
        line = f.readline()
        hdrs = {}
        while True:
            h = f.readline().decode().strip()
            if not h:
                break
            k, _, v = h.partition(":")
            hdrs[k.lower()] = v.strip()
        c = ws.WebSocket(f, f, client=True)
        return sock, c, line.decode().strip(), hdrs, key

    sock, c, status, hdrs, key = connect()
    check("101" in status and hdrs.get("sec-websocket-accept") == ws.accept_key(key), "WebSocket: рукопожатие 101, ключ принят")
    c.send_text(json.dumps({"type": "start", "voice": "demo", "format": "f32", "seed": 42}))
    k, ready = c.recv()
    ready = json.loads(ready)
    check(ready["type"] == "ready" and ready["sample_rate"] == 24000 and ready["capabilities"]["stream"]["text_in"] == "token", "WebSocket: ready с частотой и capabilities")
    for piece in ["Hello th", "ere. How a", "re you today? Fine, thanks."]:
        c.send_text(json.dumps({"type": "text", "text": piece}))
        time.sleep(0.05)
    c.send_text(json.dumps({"type": "end"}))
    pcm, done = b"", None
    while True:
        k, m = c.recv()
        if k == "binary":
            pcm += m
        else:
            j = json.loads(m)
            if j["type"] == "done":
                done = j
                break
    got = np.frombuffer(pcm, "<f4")
    check(done is not None and np.array_equal(got, want), f"WebSocket: текст порциями → тот же звук (f32, {len(got)} отсч.), done.utterances={done and done['utterances']}")
    sock.close()

    sock, c, *_ = connect()
    c.send_text(json.dumps({"type": "start", "voice": "demo", "format": "pcm16", "commit": "manual", "seed": 42}))
    c.recv()
    c.send_text(json.dumps({"type": "set_text", "text": "Hello th"}))
    k, m = c.recv()
    j = json.loads(m)
    check(j["type"] == "set_text_result" and j["ok"] and j["committed"] == 0, "WebSocket: set_text → set_text_result")
    c.send_text(json.dumps({"type": "set_text", "text": "Hello there. How are you today? Fine, thanks."}))
    c.recv()
    c.send_text(json.dumps({"type": "end"}))
    pcm = b""
    while True:
        k, m = c.recv()
        if k == "binary":
            pcm += m
        elif json.loads(m)["type"] == "done":
            break
    got16 = np.frombuffer(pcm, "<i2").astype(np.float32) / 32767
    check(len(got16) == len(want) and float(np.abs(got16 - want).max()) < 1e-4, "WebSocket: набор + pcm16 даёт тот же звук (в пределах 16 бит)")
    sock.close()

    sock, c, *_ = connect()
    c.send_text(json.dumps({"type": "start", "voice": "nope"}))
    k, m = c.recv()
    check(json.loads(m)["type"] == "error", "WebSocket: неизвестный голос → error")
    sock.close()
    sock, c, *_ = connect()
    c.send_text(json.dumps({"type": "start", "voice": "demo", "commit": "whenever"}))
    k, m = c.recv()
    check(json.loads(m)["type"] == "error" and "commit" in json.loads(m)["message"], "WebSocket: неверный commit → error")
    sock.close()
    # обычный GET на /v1/duplex без Upgrade
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/duplex")
        check(False, "GET без Upgrade")
    except urllib.error.HTTPError as e:
        check(e.code == 400, "GET /v1/duplex без Upgrade → 400")
    httpd.shutdown()


def main(cli):
    with tempfile.TemporaryDirectory() as d:
        model = os.path.join(d, "test.tmod")
        subprocess.run([cli, "make-test-model", "-o", model], check=True, capture_output=True)
        eng = vox.Engine(model)
        voice = eng.clone_voice(ref_audio(), "Пример референса", name="v")
        text = "Hello there. How are you today? Fine, thanks."
        want = eng.synthesize(text, voice, seed=42)
        caps = eng.capabilities["stream"]
        check(caps["text_in"] == "token" and caps["audio_out"] == "frames" and set(caps["commit"]) == {"auto", "sentence", "manual"},
              f"engine.capabilities['stream'] = {caps}")

        # 1. живой вход + живой выход из разных потоков
        got = []
        with eng.duplex(voice, seed=42) as dx:
            def reader():
                for c in dx.audio():
                    got.append(c)
            th = threading.Thread(target=reader)
            th.start()
            for piece in ["Hel", "lo th", "ere. How a", "re you tod", "ay? Fine, ", "thanks."]:
                dx.write(piece)
                time.sleep(0.01)
            dx.end()
            th.join(10)
        check(not th.is_alive() and np.array_equal(np.concatenate(got), want),
              f"write() в одном потоке, audio() в другом = синтез целиком ({len(got)} кусков)")
        check(dx.stats["first_audio_s"] is not None and dx.stats["first_audio_s"] < 5, f"задержка до первого звука измерена: {dx.stats['first_audio_s']:.3f} с")
        pieces_before_end = len(got)

        # 2. звук идёт ДО конца текста (не копится до end)
        with eng.duplex(voice, seed=42) as dx:
            dx.write("Hello there. ")
            c = dx.read(timeout=5)
            check(c is not None and len(c) > 0 and not dx.done, "звук приходит, пока текст ещё не закончен")
            dx.write("How are you today? Fine, thanks.")
            tail = dx.final(timeout=10)
            check(np.array_equal(np.concatenate([c, tail]), want), "первый кусок + остаток = синтез целиком")

        # 3. набор текста → одна запись (manual): до end() звука нет, итог совпадает с целым
        with eng.duplex(voice, seed=42, commit="manual") as dx:
            for ch in text:
                dx.write(ch)
            time.sleep(0.15)
            check(dx.read(timeout=0.2) is None and dx.info()["committed_bytes"] == 0, "commit=manual: до end() в модель ничего не уходит")
            audio = dx.final(timeout=10)
        check(np.array_equal(audio, want), "набор по символам → одна запись, равна синтезу целиком")

        # 4. sentence: в модель уходят только законченные предложения
        with eng.duplex(voice, seed=42, commit="sentence") as dx:
            dx.write("Hello there. How ar")
            inf = dx.info()
            check(inf["committed_bytes"] == len("Hello there. ") and inf["pending"] == "How ar", f"commit=sentence держит недописанное предложение: {inf['committed_bytes']} / {inf['pending']!r}")
            dx.write("e you today? Fine, thanks.")
            check(np.array_equal(dx.final(timeout=10), want), "итог при sentence совпадает с синтезом целиком (AR-модель)")

        # 5. idle_ms: набор замолчал — слово дочитывается само
        with eng.duplex(voice, seed=42, commit="manual", idle_ms=60) as dx:
            dx.write("idle")
            c = dx.read(timeout=3)
            inf = dx.info()
            check(c is not None and inf["idle_flushes"] == 1, "idle_ms: после паузы накопленное коммитится само")
            dx.cancel()

        # 6. режим набора с правками
        with eng.duplex(voice, seed=42) as dx:
            r1 = dx.set_text("Hello th")
            r2 = dx.set_text("Hello there. How ar")           # дописали и «th»→«there»
            r3 = dx.set_text("Hello there. How are you today? Fine, thanks.")
            check(r1["ok"] and r2["ok"] and r3["ok"], "set_text: дописывание и правка хвоста принимаются")
            dx.end()
            check(np.array_equal(np.concatenate(list(dx.audio())), want), "после правок хвоста звук такой же, как у текста целиком")
        with eng.duplex(voice, seed=42, on_conflict="keep") as dx:
            dx.set_text("Hello there. ")
            time.sleep(0.1)
            r = dx.set_text("Jello there. ")
            check(not r["ok"] and r["spoken"].startswith("Hello"), "правка внутри озвученного: keep → ok=False, сказанное не меняется")
            dx.cancel()
        with eng.duplex(voice, seed=42, on_conflict="raise") as dx:
            dx.set_text("Hello there. ")
            time.sleep(0.1)
            try:
                dx.set_text("Jello there. ")
                check(False, "on_conflict=raise должен бросать")
            except ValueError as e:
                check("already spoken" in str(e), "on_conflict=raise: исключение с пояснением")
            dx.cancel()
        with eng.duplex(voice, seed=42, on_conflict="respeak") as dx:
            dx.set_text("Hello there. ")
            time.sleep(0.1)
            r = dx.set_text("Hello three. ")
            dx.end()
            n = sum(len(c) for c in dx.audio())
            check(r["ok"] and r.get("respeak_from") == 6 and dx.utterances == 2 and n > 0, f"on_conflict=respeak: новое высказывание с «three» ({r})")

        # 7. пауза между высказываниями закрывает фразу
        with eng.duplex(voice, seed=1, utterance_gap_ms=120) as dx:
            dx.write("one two")
            time.sleep(0.6)
            a = []
            while True:
                c = dx.read(timeout=0.1)
                if c is None:
                    break
                a.append(c)
            check(len(a) > 0 and dx.utterances == 1 and not dx.done, "utterance_gap_ms: фраза дочитана без end()")
            dx.write("three")
            dx.end()
            rest = list(dx.audio())
            check(dx.utterances == 2 and sum(len(c) for c in rest) > 0, "следующий write() открывает новое высказывание")

        # 8. отмена будит читателя, ошибки внятные
        dx = eng.duplex(voice, seed=1, commit="manual")
        dx.write("never spoken")
        res = []
        th = threading.Thread(target=lambda: res.append(list(dx.audio())))
        th.start()
        time.sleep(0.1)
        dx.cancel()
        th.join(5)
        check(not th.is_alive(), "cancel() будит блокирующий audio()")
        dx = eng.duplex(voice, seed=1)
        dx.write("x")
        dx.end()
        try:
            dx.write("late")
            check(False, "write после end должен падать")
        except vox.VoxError as e:
            check(e.status == -7, "write() после end() → VoxError(STATE)")
        list(dx.audio())
        try:
            eng.duplex(voice, commit="whenever")
            check(False, "неверный commit")
        except ValueError:
            check(True, "неверный commit → ValueError")

        # 9. asyncio
        async def acollect():
            with eng.duplex(voice, seed=42) as dx:
                dx.write(text)
                dx.end()
                return [c async for c in dx.aaudio()]
        check(np.array_equal(np.concatenate(asyncio.run(acollect())), want), "aaudio(): тот же звук в asyncio")

        # 10. политики и flush на уровне Session
        s = eng.session(voice, seed=42, commit="manual")
        s.feed("abcd")
        check(s.info["committed_bytes"] == 0 and s.info["pending"] == "abcd", "Session.info: manual копит")
        s.flush()
        check(s.info["committed_bytes"] == 4 and s.info["flushes"] == 1, "Session.flush() коммитит слово без пробела")
        s.cancel()
        # 11. CLI: vox live — текст из stdin на лету, итог = vox say
        vpath = os.path.join(d, "v.tvoice")
        voice.save(vpath)
        p = subprocess.Popen([cli, "live", "--model", model, "--voice", vpath, "-o", os.path.join(d, "live.wav"), "--seed", "42", "--float"],
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for piece in [b"Hello th", b"ere. How a", b"re you today? Fine, thanks."]:
            p.stdin.write(piece)
            p.stdin.flush()
            time.sleep(0.1)
        err = p.communicate(timeout=60)[1].decode()   # communicate() сам закрывает stdin (ручной close() на Python 3.12 даёт «flush of closed file»)
        subprocess.run([cli, "say", text, "--model", model, "--voice", vpath, "-o", os.path.join(d, "whole.wav"), "--seed", "42", "--float"], check=True, capture_output=True)
        a, _ = vox.load_wav(os.path.join(d, "live.wav"))
        b, _ = vox.load_wav(os.path.join(d, "whole.wav"))
        check(p.returncode == 0 and np.array_equal(a, b), "vox live: текст из stdin порциями → та же запись, что у vox say")
        # raw в stdout идёт по мере готовности
        p = subprocess.Popen([cli, "live", "--model", model, "--voice", vpath, "--raw", "--seed", "42", "--commit", "manual", "--idle-ms", "80"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.stdin.write(b"abc def")
        p.stdin.flush()
        first = p.stdout.read(4 * 4096)      # звук пришёл по idle, хотя stdin ещё открыт
        check(len(first) == 4 * 4096, "vox live --raw --idle-ms: звук идёт в stdout, пока stdin ещё открыт")
        p.stdin.close()
        rest = p.stdout.read()
        p.wait(timeout=30)
        check(p.returncode == 0 and len(rest) > 0, "vox live --raw: после закрытия stdin запись дочитана")
        ws_check(eng, voice, d, text, want)
    print("duplex: " + ("ALL OK" if fails == 0 else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "build", "vox")))
