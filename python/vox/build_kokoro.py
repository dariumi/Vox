"""Kokoro (hexgrad/Kokoro-82M and compatible StyleTTS2 + iSTFTNet checkpoints) → .tmod.

    python -m vox.build_kokoro --model Kokoro-82M/ -o kokoro.tmod [--dtype bf16] [--voices Kokoro-82M/voices]

--model    folder with config.json and kokoro-v1_0.pth (or the two files: --config, --ckpt)
--voices   folder with voice packs (*.pt, [510, 1, 256]); default: <model>/voices. Every pack becomes a named speaker.
--no-g2p   do not embed the English lexicons of misaki (then the engine needs phonemes or the Python frontend)

What is stored: every weight of KModel (weight_norm folded into plain weights), the phoneme vocabulary, the voice packs
and, for the built-in English G2P, misaki's us/gb gold + silver lexicons (when the `misaki` package is importable or
--lexicons points to its data folder)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .build import Emitter
from .formats import LazyBlob, write_tmod

PARTS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


def _strip(k: str) -> str:
    while k.startswith("module."):
        k = k[7:]
    return k


def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """{part.name: float32 array}, weight_norm (weight_g, weight_v) folded into .weight."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not all(p in sd for p in PARTS):
        raise SystemExit(f"{path}: expected the Kokoro layout {{{', '.join(PARTS)}}}, got {list(sd)[:8]}")
    out: Dict[str, np.ndarray] = {}
    for part in PARTS:
        raw = {_strip(k): v.detach().float().numpy() for k, v in sd[part].items() if hasattr(v, "shape")}
        for k, v in raw.items():
            if k.endswith(".weight_g") or k.endswith(".parametrizations.weight.original0"):
                continue
            if k.endswith(".weight_v") or k.endswith(".parametrizations.weight.original1"):
                base = k[: -len(".weight_v")] if k.endswith(".weight_v") else k[: -len(".parametrizations.weight.original1")]
                g = raw[base + (".weight_g" if k.endswith(".weight_v") else ".parametrizations.weight.original0")]
                axes = tuple(range(1, v.ndim))
                nrm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
                out[f"{part}.{base}.weight"] = (v.astype(np.float64) * (g.astype(np.float64) / nrm)).astype(np.float32)
            else:
                out[f"{part}.{k}"] = v
    return out


def lexicon_dir(explicit: str | None):
    if explicit:
        return Path(explicit)
    try:
        import misaki.data
        return Path(misaki.data.__file__).parent
    except Exception:  # noqa: BLE001
        return None


def weights_sha256(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        fh = hashlib.sha256()
        with open(p, "rb") as fp:
            for block in iter(lambda: fp.read(1 << 22), b""):
                fh.update(block)
        h.update(f"{Path(p).name}:{fh.hexdigest()}\n".encode())
    return h.hexdigest()


def build(config: str, ckpt: str, out: str, voices: str | None = None, dtype: str = "bf16", g2p: bool = True,
          lexicons: str | None = None, variant: str = "", sha: bool = True):
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    W = load_checkpoint(ckpt)
    em = Emitter(dtype)
    n_mats = 0
    for k, v in W.items():
        is_lstm_w = ".weight_ih_l0" in k or ".weight_hh_l0" in k
        if v.ndim == 2 and (k.endswith(".weight") or is_lstm_w) and "LayerNorm" not in k and "layer_norm" not in k:
            em.mat(k, (lambda v=v: v), list(v.shape))          # linear / LSTM / embedding tables in the chosen dtype
            n_mats += 1
        elif v.ndim == 3 and k.endswith(".weight") and v.size >= 4096:
            em.mat(k, (lambda v=v: v), list(v.shape))          # convolutions (converted to float on load)
        else:
            em.vec(k, (lambda v=v: v), list(v.shape))          # biases, norms, snake alphas, small kernels
    # voice packs → named speakers
    names = []
    vdir = Path(voices) if voices else Path(ckpt).parent / "voices"
    if vdir.is_dir():
        import torch
        for p in sorted(vdir.glob("*.pt")):
            pack = torch.load(p, map_location="cpu", weights_only=True).float().numpy()
            pack = pack.reshape(pack.shape[0], -1)
            if pack.shape[1] != 2 * cfg["style_dim"]:
                print(f"warning: {p.name}: pack shape {pack.shape} is not [N, {2 * cfg['style_dim']}], skipped", file=sys.stderr)
                continue
            em.vec("voice." + p.stem.lower(), (lambda a=pack: a), list(pack.shape))
            names.append(p.stem.lower())
    # misaki lexicons for the built-in English G2P
    g2p_cfg = {}
    if g2p:
        d = lexicon_dir(lexicons)
        if d is None or not (d / "us_gold.json").exists():
            print("note: misaki lexicons not found (pip install misaki, or --lexicons DIR): the engine will need phonemes "
                  "or the Python frontend for text", file=sys.stderr)
        else:
            for f in ("us_gold", "us_silver", "gb_gold", "gb_silver"):
                p = d / (f + ".json")
                if p.exists():
                    raw = np.frombuffer(p.read_bytes(), dtype=np.uint8).copy()
                    em.blobs["g2p." + f] = LazyBlob("U8", [raw.size], (lambda r=raw: r))
                    g2p_cfg[f] = True
    ist = cfg["istftnet"]
    manifest = {
        "format": "1.0",
        "model": {"family": "kokoro", "variant": variant or Path(ckpt).stem, "type": "tts",
                  "weights_sha256": weights_sha256([ckpt]) if sha else ""},
        "target": {"backend": "cpu", "arch": "any", "features": []},
        "audio": {"sample_rate": 24000, "samples_per_frame": 600, "frame_rate": 40.0, "num_codebooks": 1, "codebook_size": 1},
        "stages": [{"name": "prosody", "kind": "duration_predictor"}, {"name": "decoder", "kind": "vocoder"}],
        "precision": {"weights": dtype, "quant": "none"},
        "generation": {"speed": 1.0},
        "kokoro": {
            "vocab": cfg["vocab"], "n_token": cfg["n_token"], "hidden_dim": cfg["hidden_dim"], "style_dim": cfg["style_dim"],
            "n_layer": cfg["n_layer"], "max_dur": cfg["max_dur"], "text_encoder_kernel_size": cfg["text_encoder_kernel_size"],
            "plbert": {**cfg["plbert"], "embedding_size": int(W["bert.embeddings.word_embeddings.weight"].shape[1]), "layer_norm_eps": 1e-12},
            "istftnet": ist, "voices": names, "g2p": g2p_cfg, "context_length": cfg["plbert"].get("max_position_embeddings", 512),
        },
    }
    write_tmod(out, manifest, dict(em.blobs))
    print(f"{out}: {len(em.blobs)} blobs, {os.path.getsize(out) / 2**20:.1f} MiB; {len(names)} voice(s)"
          f"{'' if not g2p_cfg else ', English G2P lexicons: ' + ', '.join(g2p_cfg)}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="folder with config.json and the .pth checkpoint")
    ap.add_argument("--config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--voices", default="")
    ap.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"])
    ap.add_argument("--no-g2p", action="store_true")
    ap.add_argument("--lexicons", default="", help="folder with misaki's us_gold.json, us_silver.json, gb_gold.json, gb_silver.json")
    ap.add_argument("--variant", default="")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    m = Path(a.model) if a.model else None
    config = a.config or (str(m / "config.json") if m else "")
    ckpt = a.ckpt
    if not ckpt and m:
        cands = sorted(m.glob("*.pth"))
        if not cands:
            raise SystemExit(f"{m}: no .pth checkpoint")
        ckpt = str(cands[0])
    if not config or not ckpt:
        raise SystemExit("give --model DIR or --config and --ckpt")
    build(config, ckpt, a.out, a.voices or (str(m / "voices") if m else None), a.dtype, not a.no_g2p, a.lexicons or None,
          a.variant, sha=not a.no_hash)


if __name__ == "__main__":
    main()
