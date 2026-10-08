"""Достаёт словарь и merges Qwen2 из vocab-only GGUF (llama.cpp/models/ggml-vocab-qwen2.gguf)
и собирает из них минимальный .tmod для проверки токенизатора."""
import struct, sys, json
import numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "python"))

def read_gguf_kv(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
    def rd(fmt): n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))
    def rstr(): (n,) = rd("<Q"); return f.read(n).decode("utf-8")
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            (et,) = rd("<I"); (n,) = rd("<Q"); return [rval(et) for _ in range(n)]
        fm = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}[t]
        return rd(fm)[0]
    kv = {}
    for _ in range(nkv):
        k = rstr(); (t,) = rd("<I"); kv[k] = rval(t)
    return kv

def make_tmod(gguf, out):
    from vox.formats import write_tmod
    kv = read_gguf_kv(gguf)
    toks, merges, types = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.merges"], kv.get("tokenizer.ggml.token_type", [])
    vocab = "\n".join(f"{i}\t{t}" for i, t in enumerate(toks))
    specials = [t for t, ty in zip(toks, types) if ty == 3] or [t for t in toks if t.startswith("<|") and t.endswith("|>")]
    write_tmod(out, {"model": {"family": "tok-test"}, "tokenizer": {"specials": specials}},
               {"tokenizer.vocab": np.frombuffer(vocab.encode(), np.uint8),
                "tokenizer.merges": np.frombuffer("\n".join(merges).encode(), np.uint8)})
    return toks, merges, specials

if __name__ == "__main__":
    toks, merges, sp = make_tmod(sys.argv[1], sys.argv[2])
    print(len(toks), "tokens", len(merges), "merges", len(sp), "specials", sp[:6])
