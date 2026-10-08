"""vox build-f5: чекпойнт F5-TTS + вокодер Vocos → .tmod семейства "f5-tts".

    python -m vox.build_f5 --ckpt F5TTS_v1_Base/model_1250000.safetensors --vocab F5TTS_v1_Base/vocab.txt \\
        --vocos vocos-mel-24khz/ --dtype bf16 --quant q8 -o f5.tmod

--ckpt     model_*.safetensors или .pt (ключи "ema_model.transformer.*" либо "transformer.*"; для .pt берётся ema_model_state_dict)
--vocab    vocab.txt модели (номер строки = номер токена)
--vocos    папка charactr/vocos-mel-24khz (pytorch_model.bin) либо один файл .safetensors/.bin с весами
--tokenizer pinyin (по умолчанию, как в F5TTS_v1_Base/Emilia) | char
Размеры (dim, число слоёв, text_dim…) выводятся из самих весов; dim_head у F5 всегда 64 (--dim-head для нестандартных).
Квантование --quant q8 затрагивает только линейные слои трансформерных блоков DiT (cols кратно 32).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from .build import Emitter, STFile, quantize_q8
from .formats import LazyBlob, write_tmod


# ---------------- чтение весов ----------------
class Source:
    """name → ленивый numpy float32."""

    def __init__(self):
        self.get: Dict[str, Callable[[], np.ndarray]] = {}
        self.shape: Dict[str, list] = {}

    def add(self, name, shape, fn):
        self.get[name] = fn
        self.shape[name] = list(shape)

    def __contains__(self, k):
        return k in self.get


def _strip(k: str) -> str:
    for p in ("module.", "_orig_mod.", "ema_model.", "model."):
        while k.startswith(p):
            k = k[len(p):]
    return k


def load_weights(path: str, prefix: str = "") -> Source:
    src = Source()
    p = Path(path)
    if p.suffix == ".safetensors":
        st = STFile(p)
        for k in st.keys():
            kk = _strip(k)
            if prefix and not kk.startswith(prefix):
                continue
            src.add(kk[len(prefix):], st.shape(k), (lambda k=k: st.get(k)))
        return src
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("ema_model_state_dict", "model_state_dict", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        kk = _strip(k)
        if prefix and not kk.startswith(prefix):
            continue
        t = v.detach().float().numpy()
        src.add(kk[len(prefix):], t.shape, (lambda t=t: t))
    return src


# ---------------- банк mel-фильтров ----------------
def mel_filterbank(sr=24000, n_fft=1024, n_mels=100, f_min=0.0, f_max=None) -> np.ndarray:
    """[n_mels][n_fft/2+1] — torchaudio.functional.melscale_fbanks(norm=None, mel_scale="htk"), транспонированный."""
    f_max = f_max or sr / 2.0
    try:
        import torch
        import torchaudio.functional as AF
        fb = AF.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sr, None, "htk")
        return np.ascontiguousarray(fb.numpy().T.astype(np.float32))
    except Exception:  # noqa: BLE001 — нет torchaudio: считаем по той же формуле в numpy
        def h2m(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def m2h(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        n_freqs = n_fft // 2 + 1
        all_f = np.linspace(0, sr // 2, n_freqs)
        f_pts = m2h(np.linspace(h2m(f_min), h2m(f_max), n_mels + 2))
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[None, :] - all_f[:, None]
        down = -slopes[:, :-2] / f_diff[:-1]
        up = slopes[:, 2:] / f_diff[1:]
        return np.ascontiguousarray(np.maximum(0.0, np.minimum(down, up)).T.astype(np.float32))


def pinyin_table() -> Dict[str, str]:
    """иероглиф → слог с тоном (TONE3), как lazy_pinyin(c, style=TONE3) без контекста слова."""
    from pypinyin import Style, lazy_pinyin
    out = {}
    for cp in range(0x4E00, 0xA000):
        c = chr(cp)
        s = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)[0]
        if s != c:
            out[c] = s
    return out


def read_vocab(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        data = f.read()
    lines = data.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def weights_sha256(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and f.suffix in (".safetensors", ".pt", ".bin", ".pth"):
                fh = hashlib.sha256()
                with open(f, "rb") as fp:
                    for block in iter(lambda: fp.read(1 << 22), b""):
                        fh.update(block)
                h.update(f"{f.name}:{fh.hexdigest()}\n".encode())
    return h.hexdigest()


def find_vocos_weights(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    for n in ("pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"):
        if (p / n).exists():
            return str(p / n)
    raise SystemExit(f"{path}: no pytorch_model.bin / model.safetensors found")


BLOCK_LINEAR = re.compile(r"^transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v)\.weight|attn\.to_out\.0\.weight|attn_norm\.linear\.weight|ff\.ff\.0\.0\.weight|ff\.ff\.2\.weight)$")


def build(ckpt: str, vocab: str, vocos: str, out: str, dtype: str = "bf16", quant: str = "", tokenizer: str = "pinyin",
          variant: str = "", dim_head: int = 64, mel: dict | None = None, pinyin: bool = True, sha: bool = True, defaults: dict | None = None):
    dit = load_weights(ckpt, "transformer.")
    if not dit.get:
        raise SystemExit(f"{ckpt}: no 'transformer.*' tensors found")
    voc = load_weights(find_vocos_weights(vocos))
    voc.get = {k: v for k, v in voc.get.items() if k.startswith(("backbone.", "head.out."))}
    voc.shape = {k: v for k, v in voc.shape.items() if k in voc.get}

    dim = dit.shape["time_embed.time_mlp.2.weight"][0]
    depth = 1 + max(int(m.group(1)) for k in dit.get if (m := re.match(r"transformer_blocks\.(\d+)\.", k)))
    text_rows, text_dim = dit.shape["text_embed.text_embed.weight"]
    conv_layers = 1 + max((int(m.group(1)) for k in dit.get if (m := re.match(r"text_embed\.text_blocks\.(\d+)\.", k))), default=-1)
    inner = dit.shape["transformer_blocks.0.attn.to_q.weight"][0]
    heads = inner // dim_head
    ff_mult = dit.shape["transformer_blocks.0.ff.ff.0.0.weight"][0] // dim
    mel_dim = dit.shape["proj_out.weight"][0]
    conv_mult = (dit.shape["text_embed.text_blocks.0.pwconv1.weight"][0] // text_dim) if conv_layers else 2
    if any(k.startswith("long_skip_connection") for k in dit.get):
        raise SystemExit("long_skip_connection is not supported yet")
    if any(".q_norm." in k for k in dit.get):
        raise SystemExit("qk_norm is not supported yet")
    if dit.shape["input_embed.proj.weight"][1] != 2 * mel_dim + text_dim:
        raise SystemExit("unexpected input_embed.proj shape")
    vocab_list = read_vocab(vocab)
    if len(vocab_list) + 1 != text_rows:
        print(f"warning: vocab has {len(vocab_list)} entries but the embedding has {text_rows} rows (expected vocab+1)", file=sys.stderr)

    mel = {"sample_rate": 24000, "n_fft": 1024, "hop": 256, "win": 1024, "n_mels": mel_dim, **(mel or {})}
    vdim = voc.shape["backbone.embed.weight"][0]
    vinter = voc.shape["backbone.convnext.0.pwconv1.weight"][0]
    vlayers = 1 + max(int(m.group(1)) for k in voc.get if (m := re.match(r"backbone\.convnext\.(\d+)\.", k)))
    vout = voc.shape["head.out.weight"][0]
    vocos_cfg = {"dim": vdim, "intermediate": vinter, "layers": vlayers, "n_fft": vout - 2, "hop": mel["hop"]}

    em = Emitter(dtype)
    for k in dit.get:
        name = "dit." + k
        shp = dit.shape[k]
        fn = dit.get[k]
        if k.endswith(".weight") and len(shp) == 2:
            q8 = quant == "q8" and BLOCK_LINEAR.match(k) is not None
            em.mat(name, fn, shp, q8=q8)
        else:  # bias, свёртки, нормы, GRN — всегда F32
            em.vec(name, fn, shp)
    for k in voc.get:
        name = "vocos." + k
        shp = voc.shape[k]
        fn = voc.get[k]
        if k.endswith(".weight") and len(shp) == 2:
            em.mat(name, fn, shp)
        else:
            em.vec(name, fn, shp)
    fb = mel_filterbank(mel["sample_rate"], mel["n_fft"], mel["n_mels"])
    em.vec("mel.fb", lambda: fb, fb.shape)

    tok = {"type": tokenizer, "vocab": vocab_list}
    if tokenizer == "pinyin" and pinyin:
        try:
            tok["pinyin"] = pinyin_table()
        except ImportError:
            print("warning: pypinyin is not installed; Chinese characters will not be converted to pinyin", file=sys.stderr)

    gen = {"nfe_step": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "speed": 1.0, "cross_fade_duration": 0.15, "target_rms": 0.1, "use_epss": True}
    gen.update(defaults or {})
    manifest = {
        "format": "1.0",
        "model": {"family": "f5-tts", "variant": variant or Path(ckpt).parent.name or "f5-tts", "type": "tts",
                  "weights_sha256": weights_sha256([ckpt, vocos]) if sha else ""},
        "target": {"backend": "cpu", "arch": "any", "features": []},
        "audio": {"sample_rate": mel["sample_rate"], "samples_per_frame": mel["hop"], "frame_rate": mel["sample_rate"] / mel["hop"],
                  "num_codebooks": 1, "codebook_size": 1},
        "stages": [{"name": "dit", "kind": "flow_matching"}, {"name": "vocos", "kind": "vocoder"}],
        "precision": {"weights": dtype, "quant": quant or "none"},
        "generation": gen,
        "f5": {"dim": dim, "depth": depth, "heads": heads, "dim_head": dim_head, "ff_mult": ff_mult, "text_dim": text_dim,
               "text_rows": text_rows, "mel_dim": mel_dim, "conv_layers": conv_layers, "conv_mult": conv_mult,
               "text_mask_padding": True, "qk_norm": None, "mel": mel, "vocos": vocos_cfg, "tokenizer": tok},
    }
    write_tmod(out, manifest, dict(em.blobs))
    print(f"{out}: {len(em.blobs)} blobs, {os.path.getsize(out) / 2**20:.1f} MiB", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--vocos", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"])
    ap.add_argument("--quant", default="", choices=["", "q8"])
    ap.add_argument("--tokenizer", default="pinyin", choices=["pinyin", "char"])
    ap.add_argument("--variant", default="")
    ap.add_argument("--dim-head", type=int, default=64)
    ap.add_argument("--no-pinyin-table", action="store_true")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    build(a.ckpt, a.vocab, a.vocos, a.out, a.dtype, a.quant, a.tokenizer, a.variant, a.dim_head, pinyin=not a.no_pinyin_table, sha=not a.no_hash)


if __name__ == "__main__":
    main()
