"""Сверка C++ токенизатора с HF tokenizers на словаре Qwen2 (нужны tokenizers, regex и vocab gguf)."""
import json, os, random, subprocess, sys, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tok_gguf import read_gguf_kv, make_tmod
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex, AddedToken

PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

def hf_tokenizer(gguf):
    kv = read_gguf_kv(gguf)
    toks, merges = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.merges"]
    vocab = {t: i for i, t in enumerate(toks)}
    tk = Tokenizer(models.BPE(vocab, [tuple(m.split(" ", 1)) for m in merges]))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT), "isolated", False),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([AddedToken(t, special=True) for t in ("<|endoftext|>", "<|im_start|>", "<|im_end|>")])
    return tk

def cpp_encode(cli, tmod, texts):
    inp = "\n".join(json.dumps(t) for t in texts) + "\n"
    out = subprocess.run([cli, "tokenize", "--model", tmod], input=inp, capture_output=True, text=True, check=True).stdout
    return [json.loads(l) for l in out.splitlines()]

def corpus(n=3000, seed=1):
    rnd = random.Random(seed)
    fixed = ["Hello, world!", "Привет! Это проверка движка синтеза речи.", "Hello,world 12345 don't I'll YOU'RE",
             "  leading and trailing   ", "a\n\nb\r\n\r\nc", "tab\there", "x  \n  y", "...!!!???", " ,hello", "\n", " ",
             "<|im_start|>assistant\nПривет<|im_end|>\n<|im_start|>assistant\n", "emoji 😀 test", "日本語のテキスト", "mixed Русский and English 2026-10-06",
             "it's we've they'd I'M SHE'S", "a  b   c    d", "end.\n", "word!\nnext", "price: $1,250.00 (14%)", "ﬁ ligature ǅ titlecase ² superscript ½",
             "\u00a0nbsp\u2003em space", "x\u2028y", "ß ǆ İ i̇"]
    pools = ["abcdefghijklmnopqrstuvwxyz", "ABCXYZ", "абвгдежзийклмнопрстуфхцчшщъыьэюя", "0123456789", " ", " ", " ", "\n", "\r\n",
             ".,!?;:'\"()-_/\\", "αβγδ", "日本語中文", "😀🎉", "\u00a0\u3000\t"]
    out = list(fixed)
    for _ in range(n):
        out.append("".join(rnd.choice(rnd.choice(pools)) for _ in range(rnd.randint(1, 40))))
    return [unicodedata.normalize("NFC", t) for t in out]

def main(cli, gguf):
    import tempfile
    tmod = os.path.join(tempfile.gettempdir(), "vox_tok_test.tmod")
    make_tmod(gguf, tmod)
    hf = hf_tokenizer(gguf)
    texts = corpus()
    ours = cpp_encode(cli, tmod, texts)
    bad = 0
    for t, o in zip(texts, ours):
        ref = hf.encode(t, add_special_tokens=False).ids
        if ref != o:
            bad += 1
            if bad <= 8: print("MISMATCH", repr(t), "\n  hf :", ref, "\n  ours:", o)
    print(f"{len(texts) - bad}/{len(texts)} identical")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    gg = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("VOX_QWEN2_GGUF", "")
    if not gg or not os.path.exists(gg):
        print("SKIP: нужен словарь Qwen2 gguf: передайте путь вторым аргументом или VOX_QWEN2_GGUF=...\n"
              "  curl -L -o qwen2.gguf https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/ggml-vocab-qwen2.gguf")
        sys.exit(0)
    main(sys.argv[1] if len(sys.argv) > 1 else "build/vox", gg)
