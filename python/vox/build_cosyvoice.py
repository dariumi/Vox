"""CosyVoice2 (FunAudioLLM/CosyVoice2-0.5B) → .tmod.

    python -m vox.build_cosyvoice --model CosyVoice2-0.5B/ -o cosyvoice2.tmod [--dtype bf16] [--flow-dtype f32]

--model        the folder of the release: cosyvoice2.yaml, llm.pt, flow.pt, hift.pt, campplus.onnx,
               speech_tokenizer_v2.onnx, CosyVoice-BlankEN/ (Qwen2 config + tokenizer)
--dtype        the LLM's matrices (bf16 default; f16, f32)
--flow-dtype   flow and HiFT matrices (f32 default — they are small and the vocoder is sensitive)
--quant q8     the LLM's linear layers in 8 bits (blocks of 32, like Qwen3-TTS q8): about twice as fast per token on the CPU
--no-clone     leave out the speaker encoder (CAM++) and the speech tokenizer: the model then only speaks with voices
               made by another conversion of the same model (profiles carry the features)

What is stored: the LLM (Qwen2 backbone, speech embeddings, decoder head), the flow (token embedding, Conformer encoder,
U-Net estimator, the fixed noise CausalConditionalCFM draws with seed 0), HiFT (weight_norm folded), the frontend nets —
CAM++ from campplus.onnx (its graph is walked in execution order; Conv + BatchNorm pairs the exporter fused stay fused) and
S3TokenizerV2 from speech_tokenizer_v2.onnx (through s3tokenizer.utils.onnx2torch) — the mel / fbank / Whisper filterbanks
and the Qwen2 tokenizer with CosyVoice2's special tokens.
Needs: torch, pyyaml; for cloning also onnx (and onnxruntime to self-check CAM++), s3tokenizer."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .build import Emitter
from .build_kokoro import weights_sha256
from .formats import write_tmod

# CosyVoice2Tokenizer.special_tokens (additional_special_tokens, in order)
COSY2_SPECIALS = ['<|im_start|>', '<|im_end|>', '<|endofprompt|>', '[breath]', '<strong>', '</strong>', '[noise]', '[laughter]',
                  '[cough]', '[clucking]', '[accent]', '[quick_breath]', "<laughter>", "</laughter>", "[hissing]", "[sigh]",
                  "[vocalized-noise]", "[lipsmack]", "[mn]"]

# emotions of the engine's markup → CosyVoice2 instruct2 phrases (the model was trained on Chinese instructions)
INSTRUCT_EMOTIONS = {
    "happy": "用开心的语气说这句话", "sad": "用伤心的语气说这句话", "angry": "用生气的语气说这句话",
    "excited": "用兴奋的语气说这句话", "calm": "用平静的语气说这句话", "whisper": "用轻声细语的方式说这句话",
    "shout": "用大声喊叫的方式说这句话", "fear": "用恐惧的语气说这句话", "surprised": "用惊讶的语气说这句话",
    "serious": "用严肃的语气说这句话", "tender": "用温柔的语气说这句话", "tired": "用疲惫的语气说这句话",
    "sarcastic": "用讽刺的语气说这句话",
}


# ---------------------------------------------------------------- yaml (hyperpyyaml tags kept as plain mappings)
def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml (to read cosyvoice2.yaml)")

    class Loader(yaml.SafeLoader):
        pass

    def tagged(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            d = loader.construct_mapping(node, deep=True)
            d["__tag__"] = suffix
            return d
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        v = loader.construct_scalar(node)
        return {"__tag__": suffix, "__value__": v}

    Loader.add_multi_constructor("!", tagged)
    with open(path, encoding="utf-8") as f:
        raw = yaml.load(f, Loader=Loader)

    def resolve(x):
        if isinstance(x, dict):
            if x.get("__tag__") == "ref" and "__value__" in x:
                v = str(x["__value__"]).strip()
                if v.startswith("<") and v.endswith(">") and v.count("<") == 1:
                    return resolve(raw.get(v[1:-1]))
                return v
            return {k: resolve(v) for k, v in x.items()}
        if isinstance(x, list):
            return [resolve(v) for v in x]
        return x
    return resolve(raw)


def num(x, default):
    try:
        return type(default)(x) if x is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- checkpoints
def fold_weight_norm(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_g") or k.endswith(".parametrizations.weight.original0"):
            continue
        if k.endswith(".weight_v") or k.endswith(".parametrizations.weight.original1"):
            base = k[: -len(".weight_v")] if k.endswith(".weight_v") else k[: -len(".parametrizations.weight.original1")]
            g = sd[base + (".weight_g" if k.endswith(".weight_v") else ".parametrizations.weight.original0")]
            axes = tuple(range(1, v.ndim))
            nrm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
            out[base + ".weight"] = (v.astype(np.float64) * (g.astype(np.float64) / nrm)).astype(np.float32)
        else:
            out[k] = v
    return out


def torch_state(path) -> Dict[str, np.ndarray]:
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {k: v.detach().float().numpy() for k, v in sd.items() if hasattr(v, "shape")}


# ---------------------------------------------------------------- CAM++ from ONNX
def campplus_template() -> List[Tuple[str, str]]:
    """Parameterised ops of CAMPPlus.forward in execution order: ("conv" | "bn", module path)."""
    s: List[Tuple[str, str]] = []
    s += [("conv", "head.conv1"), ("bn", "head.bn1")]
    for layer in ("layer1", "layer2"):
        for i in (0, 1):
            p = f"head.{layer}.{i}."
            s += [("conv", p + "conv1"), ("bn", p + "bn1"), ("conv", p + "conv2"), ("bn", p + "bn2")]
            if i == 0:
                s += [("conv", p + "shortcut.0"), ("bn", p + "shortcut.1")]
    s += [("conv", "head.conv2"), ("bn", "head.bn2")]
    s += [("conv", "xvector.tdnn.linear"), ("bn", "xvector.tdnn.nonlinear.batchnorm")]
    for b, n in ((1, 12), (2, 24), (3, 16)):
        for i in range(1, n + 1):
            p = f"xvector.block{b}.tdnnd{i}."
            s += [("bn", p + "nonlinear1.batchnorm"), ("conv", p + "linear1"), ("bn", p + "nonlinear2.batchnorm"),
                  ("conv", p + "cam_layer.linear_local"), ("conv", p + "cam_layer.linear1"), ("conv", p + "cam_layer.linear2")]
        s += [("bn", f"xvector.transit{b}.nonlinear.batchnorm"), ("conv", f"xvector.transit{b}.linear")]
    s += [("bn", "xvector.out_nonlinear.batchnorm"), ("conv", "xvector.dense.linear"), ("bn", "xvector.dense.nonlinear.batchnorm")]
    return s


def campplus_from_onnx(path: str) -> Dict[str, np.ndarray]:
    """{"<module>.weight" / ".bias": conv, "<module>.scale" / ".shift": batch norm} in the layout of cosyvoice_front.cpp."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    for n in m.graph.node:   # weights held in Constant nodes
        if n.op_type == "Constant":
            for at in n.attribute:
                if at.name == "value":
                    init[n.output[0]] = numpy_helper.to_array(at.t)
    ops = []
    for n in m.graph.node:
        if n.op_type == "Conv":
            w = init.get(n.input[1])
            b = init.get(n.input[2]) if len(n.input) > 2 and n.input[2] else None
            if w is None:
                raise SystemExit(f"{path}: Conv {n.name} has no constant weight")
            ops.append(("conv", w, b))
        elif n.op_type == "BatchNormalization":
            eps = 1e-5
            for at in n.attribute:
                if at.name == "epsilon":
                    eps = at.f
            sc, bi, mean, var = (init[n.input[i]] for i in (1, 2, 3, 4))
            ops.append(("bn", sc, bi, mean, var, eps))
        elif n.op_type in ("Gemm", "MatMul", "PRelu"):
            raise SystemExit(f"{path}: unexpected {n.op_type} in the CAM++ graph (only Conv / BatchNormalization are supported)")
    out: Dict[str, np.ndarray] = {}
    k = 0
    fused = False   # the conv just read carries a bias it does not have in PyTorch: the BatchNorm after it was folded in
    for kind, name in campplus_template():
        nxt = ops[k] if k < len(ops) else None
        if kind == "conv":
            if nxt is None or nxt[0] != "conv":
                raise SystemExit(f"{path}: the graph does not follow CAMPPlus (expected a Conv for {name})")
            out[name + ".weight"] = np.asarray(nxt[1], np.float32)
            if nxt[2] is not None:
                out[name + ".bias"] = np.asarray(nxt[2], np.float32)
            fused = nxt[2] is not None and not name.endswith(("cam_layer.linear1", "cam_layer.linear2"))
            k += 1
        else:
            was_fused, fused = fused, False
            if not was_fused and nxt is not None and nxt[0] == "bn":
                _, sc, bi, mean, var, eps = nxt
                s = (np.asarray(sc, np.float64) / np.sqrt(np.asarray(var, np.float64) + eps))
                out[name + ".scale"] = s.astype(np.float32)
                out[name + ".shift"] = (np.asarray(bi, np.float64) - np.asarray(mean, np.float64) * s).astype(np.float32)
                k += 1
            else:   # folded into the convolution before it by the exporter
                ch = out[[n for n in out if n.endswith(".weight")][-1]].shape[0]
                out[name + ".scale"] = np.ones(ch, np.float32)
                out[name + ".shift"] = np.zeros(ch, np.float32)
    if k != len(ops):
        raise SystemExit(f"{path}: {len(ops) - k} parameterised ops left over: not the CAMPPlus graph")
    return out


def campplus_numpy(W: Dict[str, np.ndarray], fb: np.ndarray) -> np.ndarray:
    """CAM++ forward in numpy on the converted weights (fb [T][80]) — the conversion's self-check against onnxruntime."""
    def bn(x, n, relu):   # x [C, ...]
        sh = (-1,) + (1,) * (x.ndim - 1)
        y = x * W[n + ".scale"].reshape(sh) + W[n + ".shift"].reshape(sh)
        return np.maximum(y, 0) if relu else y

    def conv2(x, n, sf):   # x [C, F, T]
        w = W[n + ".weight"]
        co, ci, k, _ = w.shape
        p = k // 2
        C, F, T = x.shape
        xp = np.pad(x, ((0, 0), (p, p), (p, p)))
        Fo = (F + 2 * p - k) // sf + 1
        y = np.zeros((co, Fo, T), np.float64)
        for kf in range(k):
            for kt in range(k):
                patch = xp[:, kf: kf + sf * (Fo - 1) + 1: sf, kt: kt + T]
                y += np.einsum("oc,cft->oft", w[:, :, kf, kt], patch)
        if n + ".bias" in W:
            y += W[n + ".bias"][:, None, None]
        return y

    def conv1(x, n, stride=1, pad=0, dil=1):   # x [C, T]
        w = W[n + ".weight"]
        co, ci, k = w.shape
        C, T = x.shape
        xp = np.pad(x, ((0, 0), (pad, pad)))
        To = (T + 2 * pad - dil * (k - 1) - 1) // stride + 1
        y = np.zeros((co, To), np.float64)
        for j in range(k):
            y += w[:, :, j] @ xp[:, j * dil: j * dil + stride * (To - 1) + 1: stride]
        if n + ".bias" in W:
            y += W[n + ".bias"][:, None]
        return y

    x = fb.T[None].astype(np.float64)                        # [1, F, T]
    x = bn(conv2(x, "head.conv1", 1), "head.bn1", True)
    for layer in ("layer1", "layer2"):
        for i in (0, 1):
            p = f"head.{layer}.{i}."
            sf = 2 if i == 0 else 1
            o = bn(conv2(x, p + "conv1", sf), p + "bn1", True)
            o = bn(conv2(o, p + "conv2", 1), p + "bn2", False)
            sc = bn(conv2(x, p + "shortcut.0", sf), p + "shortcut.1", False) if p + "shortcut.0.weight" in W else x
            x = np.maximum(o + sc, 0)
    x = bn(conv2(x, "head.conv2", 2), "head.bn2", True)
    x = x.reshape(-1, x.shape[-1])
    x = bn(conv1(x, "xvector.tdnn.linear", 2, 2), "xvector.tdnn.nonlinear.batchnorm", True)
    for b, n, d in ((1, 12, 1), (2, 24, 2), (3, 16, 2)):
        for i in range(1, n + 1):
            p = f"xvector.block{b}.tdnnd{i}."
            h = conv1(bn(x, p + "nonlinear1.batchnorm", True), p + "linear1")
            h = bn(h, p + "nonlinear2.batchnorm", True)
            y = conv1(h, p + "cam_layer.linear_local", 1, d, d)
            T = h.shape[1]
            seg = np.stack([h[:, s: s + 100].mean(1) for s in range(0, T, 100)], 1)
            ctx = h.mean(1, keepdims=True) + np.repeat(seg, 100, axis=1)[:, :T]
            m = 1 / (1 + np.exp(-conv1(np.maximum(conv1(ctx, p + "cam_layer.linear1"), 0), p + "cam_layer.linear2")))
            x = np.concatenate([x, y * m], 0)
        x = conv1(bn(x, f"xvector.transit{b}.nonlinear.batchnorm", True), f"xvector.transit{b}.linear")
    x = bn(x, "xvector.out_nonlinear.batchnorm", True)
    st = np.concatenate([x.mean(1), x.std(1, ddof=1)])[:, None]
    e = bn(conv1(st, "xvector.dense.linear"), "xvector.dense.nonlinear.batchnorm", False)
    return e[:, 0].astype(np.float32)


def check_campplus(path: str, W: Dict[str, np.ndarray]) -> str:
    try:
        import onnxruntime
    except ImportError:
        return "not checked (onnxruntime missing)"
    rng = np.random.RandomState(0)
    fb = rng.randn(230, 80).astype(np.float32)
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    sess = onnxruntime.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: fb[None]})[0].reshape(-1)
    mine = campplus_numpy(W, fb)
    err = float(np.max(np.abs(ref - mine)) / (np.max(np.abs(ref)) + 1e-9))
    if not err < 1e-3:
        raise SystemExit(f"{path}: the converted CAM++ does not reproduce onnxruntime (relative error {err:.2e})")
    return f"matches onnxruntime (rel {err:.1e})"


# ---------------------------------------------------------------- S3 tokenizer
def s3_weights(model_dir: Path):
    """(weights {name: array}, config) of S3TokenizerV2 from speech_tokenizer_v2.onnx or a PyTorch state dict."""
    pt = model_dir / "speech_tokenizer_v2.pt"
    onnx_p = model_dir / "speech_tokenizer_v2.onnx"
    if pt.exists():
        W = torch_state(pt)
    elif onnx_p.exists():
        try:
            from s3tokenizer.utils import onnx2torch
        except ImportError:
            raise SystemExit("pip install s3tokenizer onnx (to convert speech_tokenizer_v2.onnx), or pass --no-clone")
        W = {k: v.detach().float().numpy() for k, v in onnx2torch(str(onnx_p), None, False).items()}
    else:
        return None, None
    D = W["encoder.conv1.weight"].shape[0]
    n_mels = W["encoder.conv1.weight"].shape[1]
    layers = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("encoder.blocks."))
    cfg = {"n_audio_state": int(D), "n_audio_head": int(D // 64), "n_audio_layer": int(layers), "n_mels": int(n_mels)}
    return W, cfg


# ---------------------------------------------------------------- filterbanks
def slaney_mel(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax=None) -> np.ndarray:
    """librosa.filters.mel (htk=False, norm='slaney') in numpy."""
    fmax = sr / 2 if fmax is None else fmax

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        mels = f / (200.0 / 3)
        min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        freqs = (200.0 / 3) * m
        min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    w = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    w *= enorm[:, None]
    return w.astype(np.float32)


def kaldi_banks(n_mels=80, padded=512, sr=16000.0, low=20.0, high=0.0) -> np.ndarray:
    """torchaudio.compliance.kaldi.get_mel_banks (no VTLN), with the zero Nyquist column → [n_mels][padded/2 + 1]."""
    nyq = 0.5 * sr
    if high <= 0.0:
        high += nyq
    nfb = padded // 2
    width = sr / padded
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f, np.float32) / 700.0)   # noqa: E731
    ml, mh = float(mel(low)), float(mel(high))
    delta = (mh - ml) / (n_mels + 1)
    b = np.arange(n_mels, dtype=np.float32)[:, None]
    left = (ml + b * delta).astype(np.float32)
    center = (ml + (b + 1.0) * delta).astype(np.float32)
    right = (ml + (b + 2.0) * delta).astype(np.float32)
    m = mel(width * np.arange(nfb, dtype=np.float32))[None, :].astype(np.float32)
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return np.pad(banks, ((0, 0), (0, 1)))


def whisper_filters(n_mels=128) -> np.ndarray:
    for mod in ("s3tokenizer", "whisper"):
        try:
            m = __import__(mod)
            p = Path(m.__file__).parent / "assets" / "mel_filters.npz"
            if p.exists():
                with np.load(p) as f:
                    return np.asarray(f[f"mel_{n_mels}"], np.float32)
        except ImportError:
            pass
    return slaney_mel(16000, 400, n_mels)


# ---------------------------------------------------------------- tokenizer
def tokenizer_blobs(d: Path):
    vocab = json.load(open(d / "vocab.json", encoding="utf-8"))
    toks = {int(i): t for t, i in vocab.items()}
    literal = []
    tcfg = d / "tokenizer_config.json"
    if tcfg.exists():
        for i, v in json.load(open(tcfg, encoding="utf-8")).get("added_tokens_decoder", {}).items():
            toks[int(i)] = v["content"]
            literal.append(v["content"])
    # CosyVoice2Tokenizer: tokenizer.add_special_tokens(...) appends the ones the vocabulary lacks, in order
    have = set(toks.values())
    nxt = max(toks) + 1
    for s in COSY2_SPECIALS:
        if s not in have:
            toks[nxt] = s
            have.add(s)
            nxt += 1
        if s not in literal:
            literal.append(s)
    merges = [l.rstrip("\n") for l in open(d / "merges.txt", encoding="utf-8") if l.strip() and not l.startswith("#version")]
    vtxt = "\n".join(f"{i}\t{toks[i]}" for i in sorted(toks)).encode()
    mtxt = "\n".join(merges).encode()
    return ({"tokenizer.vocab": np.frombuffer(vtxt, np.uint8).copy(), "tokenizer.merges": np.frombuffer(mtxt, np.uint8).copy()},
            literal, {s: k for k, s in toks.items() if s in COSY2_SPECIALS})


# ---------------------------------------------------------------- build
def build(model_dir: str, out: str, dtype: str = "bf16", flow_dtype: str = "f32", clone: bool = True, variant: str = "",
          sha: bool = True, quant: str = ""):
    import torch
    md = Path(model_dir)
    y = load_yaml(str(next(iter(sorted(md.glob("cosyvoice2*.yaml"))), md / "cosyvoice2.yaml")))
    blank = md / "CosyVoice-BlankEN"
    qcfg = json.load(open(blank / "config.json"))
    llm_y, flow_y, hift_y = y.get("llm", {}), y.get("flow", {}), y.get("hift", {})
    enc_y, dec_y = flow_y.get("encoder", {}), flow_y.get("decoder", {})
    est_y, cfm_y = dec_y.get("estimator", {}), (dec_y.get("cfm_params", {}) or {}).get("content", dec_y.get("cfm_params", {}))
    samp = llm_y.get("sampling", {})
    ist = hift_y.get("istft_params", {"n_fft": 16, "hop_len": 4})
    fe = y.get("feat_extractor", {})
    channels = est_y.get("channels", [256])
    if len(channels) != 1:
        raise SystemExit(f"estimator channels {channels}: only one level is supported (CosyVoice2 has [256])")
    cv = {
        "sample_rate": num(y.get("sample_rate"), 24000),
        "llm": {"hidden_size": qcfg["hidden_size"], "num_hidden_layers": qcfg["num_hidden_layers"],
                "num_attention_heads": qcfg["num_attention_heads"], "num_key_value_heads": qcfg.get("num_key_value_heads", qcfg["num_attention_heads"]),
                "head_dim": qcfg.get("head_dim") or qcfg["hidden_size"] // qcfg["num_attention_heads"],
                "intermediate_size": qcfg["intermediate_size"], "rms_norm_eps": qcfg.get("rms_norm_eps", 1e-6),
                "rope_theta": qcfg.get("rope_theta", 1e6), "speech_token_size": num(llm_y.get("speech_token_size"), 6561),
                "top_p": num(samp.get("top_p"), 0.8), "top_k": num(samp.get("top_k"), 25), "win_size": num(samp.get("win_size"), 10),
                "tau_r": num(samp.get("tau_r"), 0.1), "min_token_text_ratio": 2.0, "max_token_text_ratio": 20.0},
        "flow": {"encoder_dim": num(enc_y.get("output_size"), 512), "encoder_heads": num(enc_y.get("attention_heads"), 8),
                 "encoder_units": num(enc_y.get("linear_units"), 2048), "encoder_blocks": num(enc_y.get("num_blocks"), 6),
                 "encoder_up_blocks": 4, "pre_lookahead_len": num(flow_y.get("pre_lookahead_len"), 3),
                 "token_mel_ratio": num(flow_y.get("token_mel_ratio"), 2), "mel_dim": num(flow_y.get("output_size"), 80),
                 "spk_embed_dim": num(flow_y.get("spk_embed_dim"), 192),
                 "estimator_channels": int(channels[0]), "estimator_heads": num(est_y.get("num_heads"), 8),
                 "estimator_head_dim": num(est_y.get("attention_head_dim"), 64), "estimator_blocks": num(est_y.get("n_blocks"), 4),
                 "estimator_mid_blocks": num(est_y.get("num_mid_blocks"), 12), "n_timesteps": 10,
                 "inference_cfg_rate": num(cfm_y.get("inference_cfg_rate"), 0.7), "t_scheduler": str(cfm_y.get("t_scheduler", "cosine"))},
        "hift": {"base_channels": num(hift_y.get("base_channels"), 512), "nb_harmonics": num(hift_y.get("nb_harmonics"), 8),
                 "n_fft": num(ist.get("n_fft"), 16), "hop_len": num(ist.get("hop_len"), 4),
                 "upsample_rates": hift_y.get("upsample_rates", [8, 5, 3]), "upsample_kernel_sizes": hift_y.get("upsample_kernel_sizes", [16, 11, 7]),
                 "resblock_kernel_sizes": hift_y.get("resblock_kernel_sizes", [3, 7, 11]),
                 "resblock_dilation_sizes": hift_y.get("resblock_dilation_sizes", [[1, 3, 5]] * 3),
                 "source_resblock_kernel_sizes": hift_y.get("source_resblock_kernel_sizes", [7, 7, 11]),
                 "source_resblock_dilation_sizes": hift_y.get("source_resblock_dilation_sizes", [[1, 3, 5]] * 3),
                 "lrelu_slope": num(hift_y.get("lrelu_slope"), 0.1), "audio_limit": num(hift_y.get("audio_limit"), 0.99)},
        "mel_n_fft": num(fe.get("n_fft"), 1920), "mel_hop": num(fe.get("hop_size"), 480),
    }
    if num(hift_y.get("sampling_rate"), 24000) != 24000 or cv["sample_rate"] != 24000:
        raise SystemExit("only the 24 kHz CosyVoice2 layout is supported")
    em = Emitter(dtype)
    femit = Emitter(flow_dtype)

    def put(e, k, v):
        if v.ndim == 2 and v.shape[0] * v.shape[1] >= 4096 and k.endswith(".weight") and "norm" not in k:
            e.mat(k, (lambda v=v: v), list(v.shape))
        elif v.ndim == 3 and v.size >= 4096 and k.endswith(".weight"):
            e.mat(k, (lambda v=v: v), list(v.shape))   # convolutions: converted to float on load
        else:
            e.vec(k, (lambda v=v: v), list(v.shape))

    # ---- LLM ----
    L = torch_state(md / "llm.pt")
    n_llm = 0
    for k, v in L.items():
        if k.startswith("llm.model.model."):
            name = "llm." + k[len("llm.model.model."):]
        elif k.startswith("llm.model.lm_head."):
            continue
        elif k.split(".")[0] in ("llm_embedding", "llm_decoder", "speech_embedding"):
            name = "llm." + k
        else:
            continue
        if v.ndim == 2 and "norm" not in name:
            # q8: the transformer's linear layers (attention and MLP projections), as for Qwen3-TTS; embeddings stay in dtype
            q8 = quant == "q8" and ".layers." in name and name.endswith("proj.weight")
            em.mat(name, (lambda v=v: v), list(v.shape), q8=q8)
        else:
            em.vec(name, (lambda v=v: v), list(v.shape))
        n_llm += 1
    if "llm.embed_tokens.weight" not in em.blobs:
        raise SystemExit(f"{md / 'llm.pt'}: no Qwen2 embeddings (llm.model.model.embed_tokens.weight)")
    # ---- flow ----
    Fw = torch_state(md / "flow.pt")
    for k, v in Fw.items():
        put(femit, "flow." + k, v)
    torch.manual_seed(0)   # CausalConditionalCFM.__init__: set_all_random_seed(0); torch.randn([1, 80, 50 * 300])
    noise = torch.randn([1, cv["flow"]["mel_dim"], 50 * 300])[0].numpy()
    femit.vec("flow.rand_noise", (lambda: noise), list(noise.shape))
    # ---- HiFT ----
    H = fold_weight_norm({(k[len("generator."):] if k.startswith("generator.") else k): v for k, v in torch_state(md / "hift.pt").items()})
    for k, v in H.items():
        put(femit, "hift." + k, v)
    # ---- frontend ----
    blobs = dict(em.blobs)
    blobs.update(femit.blobs)
    # feat_extractor of cosyvoice2.yaml: fmax 8000 in the release (null would mean sr / 2)
    fmax = fe.get("fmax", 8000) if isinstance(fe, dict) else 8000
    cv["mel_fmax"] = float(fmax) if fmax is not None else 12000.0
    front = {"front.mel_basis": slaney_mel(24000, cv["mel_n_fft"], cv["flow"]["mel_dim"], 0.0, fmax),
             "front.fbank_banks": kaldi_banks()}
    notes = []
    if clone:
        camp_p = md / "campplus.onnx"
        if camp_p.exists():
            CW = campplus_from_onnx(str(camp_p))
            notes.append("CAM++ " + check_campplus(str(camp_p), CW))
            for k, v in CW.items():
                front["camp." + k] = v.astype(np.float32)
            cv["campplus"] = True
        else:
            notes.append("no campplus.onnx: cloning disabled")
        SW, scfg = s3_weights(md)
        if SW is not None:
            for k, v in SW.items():
                front["s3." + k] = np.asarray(v, np.float32)
            cv["s3tokenizer"] = scfg
            front["front.whisper_filters"] = whisper_filters(scfg["n_mels"])
        else:
            notes.append("no speech_tokenizer_v2.onnx: cloning disabled")
    for k, v in front.items():
        blobs[k] = v
    tb, literal, ids = tokenizer_blobs(blank)
    blobs.update(tb)
    files = [md / "llm.pt", md / "flow.pt", md / "hift.pt"] + [p for p in (md / "campplus.onnx", md / "speech_tokenizer_v2.onnx") if p.exists()]
    manifest = {
        "format": "1.0",
        "model": {"family": "cosyvoice", "variant": variant or md.name.lower(), "type": "tts",
                  "weights_sha256": weights_sha256(files) if sha else ""},
        "target": {"backend": "cpu", "arch": "any", "features": []},
        "audio": {"sample_rate": 24000, "samples_per_frame": 480, "frame_rate": 50.0, "num_codebooks": 1, "codebook_size": 1},
        "stages": [{"name": "llm", "kind": "speech_token_lm"}, {"name": "flow", "kind": "flow_matching"}, {"name": "hift", "kind": "vocoder"}],
        "precision": {"weights": dtype, "flow": flow_dtype, "quant": quant or "none"},
        "generation": {"speed": 1.0},
        "tokenizer": {"specials": literal, "cosyvoice_specials": ids},
        "instruct_emotions": INSTRUCT_EMOTIONS,
        "cosyvoice": cv,
    }
    write_tmod(out, manifest, blobs)
    print(f"{out}: {len(blobs)} blobs, {os.path.getsize(out) / 2**20:.1f} MiB; LLM {qcfg['num_hidden_layers']} layers ({n_llm} tensors)"
          + "".join("; " + n for n in notes), file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the CosyVoice2-0.5B folder")
    ap.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"])
    ap.add_argument("--flow-dtype", default="f32", choices=["f32", "f16", "bf16"])
    ap.add_argument("--quant", default="", choices=["", "q8"], help="q8: the LLM's linear layers in 8 bits (faster on the CPU)")
    ap.add_argument("--no-clone", action="store_true")
    ap.add_argument("--variant", default="")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    build(a.model, a.out, a.dtype, a.flow_dtype, not a.no_clone, a.variant, sha=not a.no_hash, quant=a.quant)


if __name__ == "__main__":
    main()
