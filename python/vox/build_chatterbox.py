"""Build a .tmod from a Chatterbox release (ResembleAI/chatterbox, English):

--model   the folder with t3_cfg.safetensors, s3gen.safetensors, ve.safetensors, tokenizer.json and (optionally) conds.pt,
          the built-in voice. Any S3Gen sizes work: the dimensions are read from the weights.

Layout of the result (src/chatterbox.cpp, src/cosyvoice.cpp, src/backend_chatterbox.cpp):
  t3.*      the T3 Llama (t3.tfmr.layers.N.*), the text / speech embeddings and their learned positions, the speech head,
            the conditioning encoder (speaker projection, perceiver, emotion "exaggeration" projection);
  ve.*      the VoiceEncoder (3-layer LSTM + projection) and front.ve_mel_basis (40 bands up to 8 kHz);
  flow.*    S3Gen's flow (the networks of CosyVoice2: Conformer encoder, U-Net estimator) — the names are the same;
  hift.*    S3Gen's HiFT (mel2wav, weight norm folded);
  camp.*    S3Gen's CAM++ speaker encoder (batch norms as scale / shift);
  s3.*      S3Gen's speech tokenizer (S3TokenizerV2);
  voice.default.*  the conditionals of conds.pt;
  tokenizer.json   the text tokenizer, as is (src/hf_tokenizer.cpp reads it).
The manifest's "cosyvoice" section describes S3Gen the way build_cosyvoice.py describes CosyVoice2, with the Chatterbox
differences: noise drawn per utterance (no fixed table), Euler steps dt = t[i+1] − t[i], the first SineGen.

The official ChatterboxTTS adds an inaudible Perth watermark to everything it says; vox does not.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .build import Emitter
from .build_cosyvoice import campplus_template, fold_weight_norm, kaldi_banks, slaney_mel, whisper_filters
from .build_kokoro import weights_sha256
from .formats import write_tmod

# Emotions of the engine's markup → the "exaggeration" Chatterbox reads (0.5 neutral; higher = more expressive, faster).
EMOTION_OPTIONS = {
    "happy": {"exaggeration": 0.8, "cfg_weight": 0.3},
    "excited": {"exaggeration": 0.95, "cfg_weight": 0.3},
    "angry": {"exaggeration": 0.9, "cfg_weight": 0.3},
    "shout": {"exaggeration": 1.0, "cfg_weight": 0.3},
    "surprised": {"exaggeration": 0.85, "cfg_weight": 0.3},
    "fear": {"exaggeration": 0.7, "cfg_weight": 0.4},
    "sarcastic": {"exaggeration": 0.65, "cfg_weight": 0.4},
    "sad": {"exaggeration": 0.35, "cfg_weight": 0.5},
    "tender": {"exaggeration": 0.35, "cfg_weight": 0.5},
    "tired": {"exaggeration": 0.25, "cfg_weight": 0.5},
    "calm": {"exaggeration": 0.3, "cfg_weight": 0.5},
    "serious": {"exaggeration": 0.4, "cfg_weight": 0.5},
    "whisper": {"exaggeration": 0.25, "cfg_weight": 0.5},
}


def load_st(path: Path) -> Dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    try:
        return {k: np.asarray(v) for k, v in load_file(str(path)).items()}
    except (TypeError, ValueError):   # bf16 tensors: through torch
        from safetensors.torch import load_file as lt
        return {k: v.float().numpy() for k, v in lt(str(path)).items()}


def campplus_from_torch(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """CAMPPlus state dict → the layout of cosyvoice_front.cpp (convolutions as they are, batch norms as scale / shift;
    the affine-free ones have no weight / bias)."""
    out: Dict[str, np.ndarray] = {}
    for kind, name in campplus_template():
        if kind == "conv":
            if name + ".weight" not in sd:
                raise SystemExit(f"CAM++: no {name}.weight")
            out[name + ".weight"] = np.asarray(sd[name + ".weight"], np.float32)
            if name + ".bias" in sd:
                out[name + ".bias"] = np.asarray(sd[name + ".bias"], np.float32)
        else:
            mean = np.asarray(sd[name + ".running_mean"], np.float64)
            var = np.asarray(sd[name + ".running_var"], np.float64)
            w = np.asarray(sd.get(name + ".weight", np.ones_like(mean)), np.float64)
            b = np.asarray(sd.get(name + ".bias", np.zeros_like(mean)), np.float64)
            s = w / np.sqrt(var + 1e-5)
            out[name + ".scale"] = s.astype(np.float32)
            out[name + ".shift"] = (b - mean * s).astype(np.float32)
    return out


def count(keys, prefix: str) -> int:
    idx = [int(m.group(1)) for k in keys for m in [re.match(re.escape(prefix) + r"(\d+)\.", k)] if m]
    return 1 + max(idx) if idx else 0


def s3gen_config(S: Dict[str, np.ndarray]) -> dict:
    """The "cosyvoice" manifest section of S3Gen, from the shapes of its weights."""
    k = S.keys()
    enc_dim = int(S["flow.encoder_proj.weight"].shape[1])
    heads = int(S["flow.encoder.encoders.0.self_attn.pos_bias_u"].shape[0])
    q = S["flow.decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight"]
    est_ch = int(q.shape[1])
    head_dim = 64   # S3Token2Mel: attention_head_dim=64
    hb = "mel2wav." if any(x.startswith("mel2wav.") for x in k) else "hift."
    return {
        "sample_rate": 24000,
        "flow": {"encoder_dim": enc_dim, "encoder_heads": heads,
                 "encoder_units": int(S["flow.encoder.encoders.0.feed_forward.w_1.weight"].shape[0]),
                 "encoder_blocks": count(k, "flow.encoder.encoders."), "encoder_up_blocks": count(k, "flow.encoder.up_encoders."),
                 "pre_lookahead_len": 3, "token_mel_ratio": 2, "mel_dim": int(S["flow.encoder_proj.weight"].shape[0]),
                 "spk_embed_dim": int(S["flow.spk_embed_affine_layer.weight"].shape[1]),
                 "estimator_channels": est_ch, "estimator_heads": int(q.shape[0]) // head_dim, "estimator_head_dim": head_dim,
                 "estimator_blocks": count(k, "flow.decoder.estimator.down_blocks.0.1."),
                 "estimator_mid_blocks": count(k, "flow.decoder.estimator.mid_blocks."), "n_timesteps": 10,
                 "inference_cfg_rate": 0.7, "t_scheduler": "cosine", "dt": "diff"},
        "hift": {"base_channels": int(S[hb + "conv_pre.weight"].shape[0]) if hb + "conv_pre.weight" in S
                 else int(S[hb + "conv_pre.parametrizations.weight.original1"].shape[0]),
                 "nb_harmonics": int(S[hb + "m_source.l_linear.weight"].shape[1]) - 1,
                 "n_fft": 16, "hop_len": 4, "upsample_rates": [8, 5, 3], "upsample_kernel_sizes": [16, 11, 7],
                 "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5]] * 3,
                 "source_resblock_kernel_sizes": [7, 7, 11], "source_resblock_dilation_sizes": [[1, 3, 5]] * 3,
                 "lrelu_slope": 0.1, "audio_limit": 0.99, "sinegen": 1},
        "mel_n_fft": 1920, "mel_hop": 480, "mel_fmax": 8000.0,
    }


def build(model_dir: str, out: str, dtype: str = "bf16", flow_dtype: str = "f32", clone: bool = True, variant: str = "",
          sha: bool = True, quant: str = ""):
    md = Path(model_dir)
    for f in ("t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"):
        if not (md / f).exists():
            raise SystemExit(f"{md}: no {f} (expected the ResembleAI/chatterbox files)")
    em = Emitter(dtype)
    femit = Emitter(flow_dtype)

    def put(e, k, v):
        if v.ndim == 2 and v.shape[0] * v.shape[1] >= 4096 and k.endswith(".weight") and "norm" not in k:
            e.mat(k, (lambda v=v: v), list(v.shape))
        elif v.ndim == 3 and v.size >= 4096 and k.endswith(".weight"):
            e.mat(k, (lambda v=v: v), list(v.shape))
        else:
            e.vec(k, (lambda v=v: v), list(v.shape))

    # ---- T3 ----
    T = load_st(md / "t3_cfg.safetensors")
    if "model" in T:
        raise SystemExit("t3_cfg.safetensors holds a training checkpoint ('model'): export its state dict first")
    hidden = int(T["speech_emb.weight"].shape[1])
    layers = count(T.keys(), "tfmr.layers.")
    if layers == 0 or "tfmr.layers.0.self_attn.q_proj.weight" not in T:
        raise SystemExit("t3_cfg.safetensors: no Llama layers (tfmr.layers.N.self_attn.*); the GPT-2 T3 (Turbo) is not supported")
    qp, kp = T["tfmr.layers.0.self_attn.q_proj.weight"], T["tfmr.layers.0.self_attn.k_proj.weight"]
    head_dim = 64
    for k, v in T.items():
        if k.startswith("tfmr.embed_tokens.") or k.startswith("text_head."):
            continue   # unused: T3 feeds its own embeddings and reads only the speech head
        name = "t3." + k
        if v.ndim == 2 and "norm" not in k and v.shape[0] * v.shape[1] >= 4096:
            q8 = quant == "q8" and ".layers." in k and k.endswith("proj.weight")
            em.mat(name, (lambda v=v: v), list(v.shape), q8=q8)
        else:
            em.vec(name, (lambda v=v: v), list(v.shape))
    t3cfg = {
        "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": int(qp.shape[0]) // head_dim,
        "num_key_value_heads": int(kp.shape[0]) // head_dim, "head_dim": head_dim,
        "intermediate_size": int(T["tfmr.layers.0.mlp.gate_proj.weight"].shape[0]), "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
        "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
                         "rope_type": "llama3"},
        "text_tokens_dict_size": int(T["text_emb.weight"].shape[0]), "speech_tokens_dict_size": int(T["speech_emb.weight"].shape[0]),
        "start_text_token": 255, "stop_text_token": 0, "start_speech_token": 6561, "stop_speech_token": 6562,
        "speech_cond_prompt_len": 150, "speaker_embed_size": int(T["cond_enc.spkr_enc.weight"].shape[1]),
        "perceiver_heads": 4,
    }
    if t3cfg["text_tokens_dict_size"] != 704:
        raise SystemExit(f"text vocabulary {t3cfg['text_tokens_dict_size']}: only the English Chatterbox (704) is supported")
    # ---- S3Gen ----
    S = load_st(md / "s3gen.safetensors")
    cv = s3gen_config(S)
    flow = {k[len("flow."):]: v for k, v in S.items() if k.startswith("flow.")}
    for k, v in flow.items():
        put(femit, "flow." + k, v)
    H = fold_weight_norm({k[len("mel2wav."):]: v for k, v in S.items() if k.startswith("mel2wav.")})
    for k, v in H.items():
        put(femit, "hift." + k, v)
    blobs = dict(em.blobs)
    blobs.update(femit.blobs)
    front: Dict[str, np.ndarray] = {"front.mel_basis": slaney_mel(24000, 1920, cv["flow"]["mel_dim"], 0.0, 8000.0),
                                    "front.fbank_banks": kaldi_banks()}
    notes = []
    has_ve = (md / "ve.safetensors").exists()
    if clone:
        camp = {k[len("speaker_encoder."):]: v for k, v in S.items() if k.startswith("speaker_encoder.")}
        if camp:
            for k, v in campplus_from_torch(camp).items():
                front["camp." + k] = v
            cv["campplus"] = True
        tok = {k[len("tokenizer."):]: v for k, v in S.items() if k.startswith("tokenizer.") and not k.endswith(("_mel_filters", "window"))}
        if tok:
            for k, v in tok.items():
                front["s3." + k] = np.asarray(v, np.float32)
            D = int(tok["encoder.conv1.weight"].shape[0])
            cv["s3tokenizer"] = {"n_audio_state": D, "n_audio_head": D // 64, "n_audio_layer": count(tok.keys(), "encoder.blocks."),
                                 "n_mels": int(tok["encoder.conv1.weight"].shape[1])}
            front["front.whisper_filters"] = whisper_filters(cv["s3tokenizer"]["n_mels"])
        if not (camp and tok and has_ve):
            notes.append("cloning disabled (no CAM++ / speech tokenizer / ve.safetensors)")
    if has_ve:
        V = load_st(md / "ve.safetensors")
        for k, v in V.items():
            if k.startswith("lstm.") or k.startswith("proj."):
                front["ve." + k] = np.asarray(v, np.float32)
        front["front.ve_mel_basis"] = slaney_mel(16000, 400, 40, 0.0, 8000.0)
    for k, v in front.items():
        blobs[k] = v
    tj = (md / "tokenizer.json").read_bytes()
    blobs["tokenizer.json"] = np.frombuffer(tj, np.uint8).copy()
    ch = {"t3": t3cfg, "default_exaggeration": 0.5}
    files = [md / f for f in ("t3_cfg.safetensors", "s3gen.safetensors", "ve.safetensors", "tokenizer.json", "conds.pt") if (md / f).exists()]
    # ---- built-in voice ----
    if (md / "conds.pt").exists():
        import torch
        c = torch.load(md / "conds.pt", map_location="cpu", weights_only=True)
        t3c, gen = c["t3"], c["gen"]
        f32 = lambda x: np.ascontiguousarray(x.detach().float().numpy().reshape(-1))   # noqa: E731
        i32 = lambda x: np.ascontiguousarray(x.detach().long().numpy().reshape(-1).astype(np.int32))   # noqa: E731
        blobs["voice.default.ve"] = f32(t3c["speaker_emb"])
        blobs["voice.default.t3_tokens"] = i32(t3c["cond_prompt_speech_tokens"])
        blobs["voice.default.s3_tokens"] = i32(gen["prompt_token"])
        pf = gen["prompt_feat"].detach().float().numpy()
        blobs["voice.default.s3_mel"] = np.ascontiguousarray(pf.reshape(-1, pf.shape[-1]))
        blobs["voice.default.xvec"] = f32(gen["embedding"])
        ea = t3c.get("emotion_adv")
        if ea is not None and hasattr(ea, "numel"):
            ch["default_exaggeration"] = float(ea.reshape(-1)[0])
    manifest = {
        "format": "1.0",
        "model": {"family": "chatterbox", "variant": variant or "chatterbox-en", "type": "tts",
                  "weights_sha256": weights_sha256(files) if sha else ""},
        "target": {"backend": "cpu", "arch": "any", "features": []},
        "audio": {"sample_rate": 24000, "samples_per_frame": 480, "frame_rate": 50.0, "num_codebooks": 1, "codebook_size": 1},
        "stages": [{"name": "t3", "kind": "speech_token_lm"}, {"name": "flow", "kind": "flow_matching"}, {"name": "hift", "kind": "vocoder"}],
        "precision": {"weights": dtype, "flow": flow_dtype, "quant": quant or "none"},
        "emotion_options": EMOTION_OPTIONS,
        "chatterbox": ch,
        "cosyvoice": cv,
    }
    write_tmod(out, manifest, blobs)
    print(f"{out}: {len(blobs)} blobs, {os.path.getsize(out) / 2**20:.1f} MiB; T3 {layers} layers × {hidden}"
          + ("; built-in voice" if "voice.default.ve" in blobs else "") + "".join("; " + n for n in notes), file=sys.stderr)


def is_chatterbox(d: str) -> bool:
    p = Path(d)
    return (p / "t3_cfg.safetensors").exists() and (p / "s3gen.safetensors").exists()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the ResembleAI/chatterbox folder")
    ap.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"], help="T3 weights")
    ap.add_argument("--flow-dtype", default="f32", choices=["f32", "f16", "bf16"], help="S3Gen flow and HiFT weights")
    ap.add_argument("--quant", default="", choices=["", "q8"], help="q8: T3's linear layers in 8 bits (faster on the CPU)")
    ap.add_argument("--no-clone", action="store_true", help="leave out CAM++, the speech tokenizer (built-in voice only)")
    ap.add_argument("--variant", default="")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    build(a.model, a.out, a.dtype, a.flow_dtype, not a.no_clone, a.variant, sha=not a.no_hash, quant=a.quant)


if __name__ == "__main__":
    main()
