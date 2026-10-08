"""Чистый ctypes поверх C ABI (без установки пакета vox): синтез в stdout — сырой float32 PCM.
    python examples/python/ctypes_example.py model.tmod voice.tvoice "Привет!" > out.f32
Путь к библиотеке: VOX_LIB (по умолчанию build/libvox.so; macOS — libvox.dylib, Windows — vox.dll).
Обычно удобнее готовый пакет: `import vox; vox.Engine(...)` (см. README)."""
import ctypes as C
import os
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
lib = C.CDLL(os.environ.get("VOX_LIB") or os.path.join(here, "..", "..", "build", "libvox.so"))
lib.vox_last_error.restype = C.c_char_p
lib.vox_engine_sample_rate.restype = C.c_uint32
lib.vox_engine_sample_rate.argtypes = [C.c_void_p]
for name in ("vox_engine_close", "vox_voice_free", "vox_session_free"):
    getattr(lib, name).argtypes = [C.c_void_p]


def check(st, what):
    if st < 0:
        raise RuntimeError(f"{what}: {lib.vox_last_error().decode()}")
    return st


def main(model, voice_path, text):
    e, v, s = C.c_void_p(), C.c_void_p(), C.c_void_p()
    check(lib.vox_engine_open(model.encode(), None, C.byref(e)), "open")
    check(lib.vox_voice_load(e, voice_path.encode(), C.byref(v)), "voice")
    params = C.create_string_buffer(72)               # vox_gen_params; struct_size проставит vox_gen_params_default
    lib.vox_gen_params_default(params)
    struct.pack_into("<Q", params, 32, 42)            # seed: uint64 по смещению 32
    check(lib.vox_session_create(e, v, params, C.byref(s)), "session")
    check(lib.vox_session_feed_text(s, text.encode()), "feed")
    check(lib.vox_session_end_text(s), "end")
    buf = (C.c_float * 4096)()
    n, total = C.c_size_t(), 0
    while True:
        st = lib.vox_session_pull(s, buf, 4096, C.byref(n))
        if st != 0:
            check(st, "pull")
            break
        sys.stdout.buffer.write(bytes(buf)[: n.value * 4])
        total += n.value
    print(f"{total} samples @ {lib.vox_engine_sample_rate(e)} Hz", file=sys.stderr)
    lib.vox_session_free(s)
    lib.vox_voice_free(v)
    lib.vox_engine_close(e)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: ctypes_example.py model.tmod voice.tvoice text")
    main(*sys.argv[1:4])
