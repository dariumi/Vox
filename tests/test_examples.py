"""Примеры из examples/ (C, C++, Rust, Go, Node.js, Python ctypes) через C ABI на тестовой модели: звук каждого должен совпасть
с `vox say --seed 42` до бита. Недостающие инструментальные цепочки пропускаются.
Запуск: python tests/test_examples.py [build/vox]"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXT = "Привет, мир!"


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def wav_f32(path):
    b = open(path, "rb").read()
    i = 12
    while i < len(b):
        cid, n = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"data":
            return np.frombuffer(b[i + 8:i + 8 + n], "<f4")
        i += 8 + n + (n & 1)


def main(cli):
    build = os.path.dirname(os.path.abspath(cli))
    work = tempfile.mkdtemp(prefix="voxex_")
    model, voice, ref = (os.path.join(work, x) for x in ("m.tmod", "v.tvoice", "ref.wav"))
    assert sh([cli, "make-test-model", "-o", model])[0] == 0
    import wave
    with wave.open(ref, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        t = np.arange(24000) / 24000
        w.writeframes((np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes())
    assert sh([cli, "voice", "create", ref, "--model", model, "--text", "hello", "-o", voice])[0] == 0
    out = os.path.join(work, "cli.wav")
    assert sh([cli, "say", TEXT, "--model", model, "--voice", voice, "--seed", "42", "-o", out, "--float"])[0] == 0
    want = wav_f32(out)
    env = dict(os.environ, LD_LIBRARY_PATH=build + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""), VOX_LIB_DIR=build)
    fails = 0

    def check(name, cmd, **kw):
        nonlocal fails
        r = subprocess.run(cmd, capture_output=True, env=env, **kw)
        got = np.frombuffer(r.stdout, "<f4")
        ok = r.returncode == 0 and np.array_equal(got, want)
        print(("ok    " if ok else "FAIL  ") + f"{name}: {len(got)} samples" + ("" if ok else " — " + r.stderr.decode()[-300:]))
        fails += not ok

    inc = os.path.join(ROOT, "include")
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc:
        exe = os.path.join(work, "c")
        assert sh([cc, os.path.join(ROOT, "examples", "minimal.c"), "-I" + inc, "-L" + build, "-lvox", "-o", exe])[0] == 0
        check("C", [exe, model, voice, TEXT])
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx:
        exe = os.path.join(work, "cpp")
        rc, o = sh([cxx, "-std=c++17", os.path.join(ROOT, "examples", "minimal.cpp"), "-I" + inc, "-L" + build, "-lvox", "-o", exe])
        assert rc == 0, o
        check("C++", [exe, model, voice, TEXT])
    check("Python (ctypes)", [sys.executable, os.path.join(ROOT, "examples", "python", "ctypes_example.py"), model, voice, TEXT],
          )
    if shutil.which("cargo"):
        rc, o = sh(["cargo", "build", "--release", "-q"], cwd=os.path.join(ROOT, "examples", "rust"), env=env)
        if rc == 0:
            check("Rust", [os.path.join(ROOT, "examples", "rust", "target", "release", "vox-example"), model, voice, TEXT])
        else:
            print("FAIL  Rust build:", o[-300:]); fails += 1
    else:
        print("SKIP  Rust (no cargo)")
    if shutil.which("go"):
        exe = os.path.join(work, "vox-go")
        rc, o = sh(["go", "build", "-o", exe, "."], cwd=os.path.join(ROOT, "examples", "go"), env=dict(env, GOFLAGS="-mod=mod"))
        if rc == 0:
            check("Go (cgo)", [exe, model, voice, TEXT])
        else:
            print("FAIL  Go build:", o[-300:]); fails += 1
    else:
        print("SKIP  Go (no go)")
    nd = os.path.join(ROOT, "examples", "node")
    if shutil.which("node") and shutil.which("npm"):
        if not os.path.isdir(os.path.join(nd, "node_modules", "koffi")):
            sh(["npm", "install", "--silent"], cwd=nd)
        if os.path.isdir(os.path.join(nd, "node_modules", "koffi")):
            check("Node.js (koffi)", ["node", os.path.join(nd, "vox.mjs"), model, voice, TEXT])
        else:
            print("SKIP  Node.js (koffi did not install)")
    else:
        print("SKIP  Node.js")
    print("examples: " + ("ALL OK" if fails == 0 else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
