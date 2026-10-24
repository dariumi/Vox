"""vox build: HF-чекпойнт Qwen3-TTS (Base/CustomVoice/VoiceDesign, 12Hz) → .tmod.

    python -m vox.build --model ./Qwen3-TTS-12Hz-1.7B-Base --dtype bf16 -o model.tmod
    python -m vox.build --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --dtype f16 --quant q8 -o model.q8.tmod

Что внутри .tmod: веса talker/code predictor/декодера кодека (имена как в HF; свёртки декодера
заранее упакованы под im2col), словарь и merges токенизатора, конфигурация (manifest["qwen3"]).
Квантование: --quant q8 | q4 | q4mlp | q4full. q8 — 8 бит на все линейные слои talker/predictor; q4 — 4 бита у внимания и MLP основного talker, остальное q8; q4mlp — 4 бита только у MLP talker; q4full — всё в 4 бита (экспериментально, качество хуже).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

from .formats import LazyBlob, write_tmod

# ----- значения по умолчанию из классов конфигурации оригинала -----
TALKER_DEFAULTS = dict(vocab_size=3072, hidden_size=1024, intermediate_size=2048, num_hidden_layers=20,
                       num_attention_heads=16, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=10000,
                       num_code_groups=32, text_hidden_size=2048, text_vocab_size=151936, codec_eos_token_id=4198,
                       codec_think_id=4202, codec_nothink_id=4203, codec_think_bos_id=4204, codec_think_eos_id=4205,
                       codec_pad_id=4196, codec_bos_id=4197)
PRED_DEFAULTS = dict(vocab_size=2048, hidden_size=1024, intermediate_size=3072, num_hidden_layers=5,
                     num_attention_heads=16, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                     rope_theta=10000, num_code_groups=32)
CODEC_DEFAULTS = dict(codebook_size=2048, hidden_size=1024, latent_dim=1024, rope_theta=10000, num_attention_heads=16,
                      num_key_value_heads=16, sliding_window=72, intermediate_size=3072, layer_scale_initial_scale=0.01,
                      rms_norm_eps=1e-5, num_hidden_layers=8, num_quantizers=16, upsample_rates=[8, 5, 4, 3],
                      upsampling_ratios=[2, 2], decoder_dim=1536)
IDS_DEFAULTS = dict(im_start_token_id=151644, im_end_token_id=151645, tts_pad_token_id=151671,
                    tts_bos_token_id=151672, tts_eos_token_id=151673)

_ST_DT = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2), "I64": ("<i8", 8), "I32": ("<i4", 4),
          "I16": ("<i2", 2), "I8": ("i1", 1), "U8": ("u1", 1), "BOOL": ("?", 1)}


# ---------------- чтение safetensors (с BF16) ----------------
class STFile:
    def __init__(self, path):
        self.path = str(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def keys(self):
        return list(self.header)

    def get(self, name) -> np.ndarray:
        """Тензор как float32 (или целочисленный) numpy-массив."""
        t = self.header[name]
        b, e = t["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + b)
            raw = f.read(e - b)
        dt, _ = _ST_DT[t["dtype"]]
        a = np.frombuffer(raw, dtype=dt)
        if t["dtype"] == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        elif t["dtype"] in ("F16",):
            a = a.astype(np.float32)
        return a.reshape(t["shape"])

    def shape(self, name):
        return list(self.header[name]["shape"])


def open_checkpoint(d: Path):
    """Одиночный model.safetensors или шарды по model.safetensors.index.json → {имя: STFile}."""
    idx = d / "model.safetensors.index.json"
    out: Dict[str, STFile] = {}
    if idx.exists():
        wm = json.load(open(idx))["weight_map"]
        files = {}
        for k, fn in wm.items():
            files.setdefault(fn, STFile(d / fn))
            out[k] = files[fn]
    else:
        f = STFile(d / "model.safetensors")
        out = {k: f for k in f.keys()}
    return out


def f32_to_bf16_bytes(a: np.ndarray) -> bytes:
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype("<u2").tobytes()


def quantize_q8(w: np.ndarray):
    rows, cols = w.shape
    blocks = w.reshape(rows, cols // 32, 32)
    s = np.abs(blocks).max(-1) / 127.0
    q = np.where(s[..., None] > 0, np.rint(blocks / np.where(s[..., None] > 0, s[..., None], 1)), 0).astype(np.int8)
    return q.reshape(rows, cols), s.astype(np.float32)


def quantize_q4(w: np.ndarray):
    """Q4: блоки по 32, байт j блока = код веса j | код веса j+16 << 4, значение = (код - 8) * масштаб.
    Масштаб подбирается как в kernels.cpp / quantize_q4: перебор d = mx/-t, t из [7, 9] с шагом 0.25 (mx — значение с
    наибольшим модулем), затем уточнение методом наименьших квадратов. t = 8 — прежняя схема. Формат хранения тот же."""
    rows, cols = w.shape
    blocks = np.ascontiguousarray(w, dtype=np.float32).reshape(rows, cols // 32, 32)
    idx = np.abs(blocks).argmax(-1)
    mx = np.take_along_axis(blocks, idx[..., None], -1)[..., 0].astype(np.float32)
    best_err = np.full(mx.shape, np.inf, dtype=np.float64)
    best_d = (mx / np.float32(-8.0)).astype(np.float32)
    best_q = np.zeros(blocks.shape, dtype=np.int8)
    for k in range(9):
        t = np.float32(7.0 + 0.25 * k)
        d = (mx / -t).astype(np.float32)
        inv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
        q = (np.clip(np.floor(blocks * inv[..., None] + np.float32(8.5)), 0, 15) - 8).astype(np.int8)
        err = ((blocks.astype(np.float64) - q * d[..., None].astype(np.float64)) ** 2).sum(-1)
        m = (err < best_err) & (mx != 0)
        best_err = np.where(m, err, best_err)
        best_d = np.where(m, d, best_d)
        best_q = np.where(m[..., None], q, best_q)
    qf = best_q.astype(np.float64)
    num = (blocks.astype(np.float64) * qf).sum(-1)
    den = (qf * qf).sum(-1)
    d2 = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    err2 = ((blocks.astype(np.float64) - qf * d2[..., None]) ** 2).sum(-1)
    best_d = np.where((err2 < best_err) & (den > 0), d2, best_d).astype(np.float32)
    code = (best_q.astype(np.int16) + 8).astype(np.uint8)
    packed = (code[..., :16] | (code[..., 16:] << 4)).astype(np.uint8)
    return packed.reshape(rows, cols // 2), best_d


class Emitter:
    """Собирает блобы с ленивым вычислением."""

    def __init__(self, dtype: str):
        self.dtype = dtype
        self.blobs: Dict[str, LazyBlob] = {}

    def vec(self, name, producer, shape):
        self.blobs[name] = LazyBlob("F32", shape, lambda: np.ascontiguousarray(producer(), dtype=np.float32))

    def mat(self, name, producer, shape, dtype=None, q8=False, q4=False):
        dtype = (dtype or self.dtype).upper()
        if q4 and len(shape) == 2 and shape[1] % 32 == 0:
            cache = {}

            def q():
                if "q" not in cache:
                    cache["q"] = quantize_q4(np.asarray(producer(), dtype=np.float32))
                return cache["q"]
            self.blobs[name] = LazyBlob("U8", [shape[0], shape[1] // 2], lambda: q()[0])
            self.blobs[name + ".scale"] = LazyBlob("F32", [shape[0], shape[1] // 32], lambda: q()[1])
            return
        if q8 and len(shape) == 2 and shape[1] % 32 == 0:
            cache = {}

            def q():
                if "q" not in cache:
                    cache["q"] = quantize_q8(np.asarray(producer(), dtype=np.float32))
                return cache["q"]
            self.blobs[name] = LazyBlob("I8", shape, lambda: q()[0])
            self.blobs[name + ".scale"] = LazyBlob("F32", [shape[0], shape[1] // 32], lambda: q()[1])
            return
        if dtype == "F32":
            self.blobs[name] = LazyBlob("F32", shape, lambda: np.asarray(producer(), dtype=np.float32))
        elif dtype == "F16":
            self.blobs[name] = LazyBlob("F16", shape, lambda: np.asarray(producer(), dtype=np.float32).astype("<f2"))
        elif dtype == "BF16":
            self.blobs[name] = LazyBlob("BF16", shape, lambda: f32_to_bf16_bytes(producer()))
        else:
            raise ValueError(dtype)


def weights_sha256(model_dir: str) -> str:
    root = Path(model_dir)
    files = sorted(root.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    h = hashlib.sha256()
    for p in files:
        fh = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                fh.update(block)
        h.update(f"{p.relative_to(root).as_posix()}:{fh.hexdigest()}\n".encode())
    return h.hexdigest()


# ---------------- тензоры ----------------
# q8: все линейные слои talker/predictor, включая головы и проекции (как Q8_0 у qwentts.cpp); эмбеддинги и нормы — нет
LINEAR_Q8 = re.compile(r"(\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$)|(^talker\.codec_head\.weight$)|"
                       r"(^talker\.code_predictor\.lm_head\.\d+\.weight$)|(^talker\.code_predictor\.small_to_mtp_projection\.weight$)|"
                       r"(^talker\.text_projection\.linear_fc[12]\.weight$)")


# q4 — смешанная схема: 4 бита только у проекций внимания и MLP основного talker (там больше всего весов, и ошибка не
# копится по 15 проходам); predictor, головы и text_projection остаются в q8. q4mlp — 4 бита только у MLP talker.
# q4full — всё в 4 бита (прежнее поведение 0.8; на реальной модели KL predictor 0.2–0.4, см. CHANGELOG).
TALKER_BODY = re.compile(r"^talker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
TALKER_MLP = re.compile(r"^talker\.model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$")


def quant_kind(name: str, quant: str) -> str:
    """Какой формат у линейного тензора: "q8", "q4" или "" (не квантуется)."""
    if not quant or not LINEAR_Q8.search(name):
        return ""
    if quant == "q8":
        return "q8"
    if quant == "q4full":
        return "q4"
    if quant == "q4":
        return "q4" if TALKER_BODY.search(name) else "q8"
    if quant == "q4mlp":
        return "q4" if TALKER_MLP.search(name) else "q8"
    raise ValueError(f"unknown quant {quant!r}")


def emit_main(em: Emitter, ck: Dict[str, STFile], quant: str, tc: dict, pc: dict):
    for name, f in ck.items():
        if not name.startswith("talker."):
            continue
        shape = f.shape(name)
        get = (lambda n=name, f=f: f.get(n))
        if len(shape) == 1:
            em.vec(name, get, shape)
        elif name.endswith("_embedding.weight") or ".codec_embedding." in name or "text_embedding" in name:
            em.mat(name, get, shape)
        else:
            kind = quant_kind(name, quant)
            em.mat(name, get, shape, q8=kind == "q8", q4=kind == "q4")


def emit_codec(em: Emitter, f: STFile, codec_dtype: str, cfg: dict):
    names = [k for k in f.keys() if k.startswith("decoder.")]
    ups = len(cfg["upsampling_ratios"])
    done_cb = set()
    for name in names:
        key = name[len("decoder."):]
        out = "codec." + key
        shape = f.shape(name)
        get = (lambda n=name: f.get(n))
        if re.search(r"quantizer\..*\.vq\.layers\.\d+\._codebook\.(cluster_usage|embedding_sum)$", key):
            base = key.rsplit(".", 1)[0]
            if base in done_cb:
                continue
            done_cb.add(base)
            es, cu = "decoder." + base + ".embedding_sum", "decoder." + base + ".cluster_usage"
            sh = f.shape(es)
            em.mat("codec." + base + ".embedding",
                   lambda es=es, cu=cu: f.get(es) / np.maximum(f.get(cu), 1e-5)[:, None], sh, dtype=codec_dtype)
            continue
        if re.search(r"quantizer\..*(input_proj|project_out)", key):
            continue
        if re.search(r"quantizer\..*output_proj\.weight$", key):
            em.mat(out, lambda g=get: g()[:, :, 0], shape[:2], dtype=codec_dtype)
        elif re.search(r"(^upsample\.\d+\.0\.conv\.weight$)|(^decoder\.\d+\.block\.1\.conv\.weight$)", key):
            cin, cout, k = shape  # ConvTranspose1d [cin, cout, K] → [K*cout, cin]
            em.mat(out, lambda g=get: g().transpose(2, 1, 0).reshape(-1, g().shape[0]), [k * cout, cin], dtype=codec_dtype)
        elif "dwconv.conv.weight" in key:
            em.vec(out, lambda g=get: g()[:, 0, :], [shape[0], shape[2]])
        elif key.endswith(".weight") and len(shape) == 3:
            cout, cin, k = shape  # Conv1d [cout, cin, K] → [cout, K*cin]
            em.mat(out, lambda g=get, s0=shape[0]: g().transpose(0, 2, 1).reshape(s0, -1), [cout, k * cin], dtype=codec_dtype)
        elif len(shape) == 1:
            em.vec(out, get, shape)
        elif len(shape) == 2 and key.endswith(".weight"):
            em.mat(out, get, shape, dtype=codec_dtype)
        else:
            em.vec(out, get, shape)


# ---------------- создание голоса: энкодер спикера (ECAPA-TDNN) и энкодер Mimi ----------------
def slaney_mel_basis(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """То же, что librosa.filters.mel(htk=False, norm="slaney") — без зависимости от librosa."""
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        f_sp, min_log_hz, min_log_mel, logstep = 200.0 / 3, 1000.0, 15.0, np.log(6.4) / 27.0
        mel = f / f_sp
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mel)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f_sp, min_log_hz, min_log_mel, logstep = 200.0 / 3, 1000.0, 15.0, np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    fft_f = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def _conv_kmajor(a: np.ndarray) -> np.ndarray:
    """Conv1d [cout, cin, K] → [cout, K*cin] (k-major: строка im2col = K сдвинутых строк по cin)."""
    return np.ascontiguousarray(a.transpose(0, 2, 1).reshape(a.shape[0], -1))


def emit_speaker_encoder(em: Emitter, ck: Dict[str, STFile]) -> bool:
    names = [k for k in ck if k.startswith("speaker_encoder.")]
    if not names:
        return False
    for name in names:
        f = ck[name]
        out = "spk." + name[len("speaker_encoder."):]
        shape = f.shape(name)
        if len(shape) == 3:
            em.mat(out, lambda n=name, f=f: _conv_kmajor(f.get(n)), [shape[0], shape[1] * shape[2]], dtype="f32")
        else:
            em.vec(out, lambda n=name, f=f: f.get(n), shape)
    # мел-спектрограмма speaker encoder (qwen_tts.extract_speaker_embedding: n_fft 1024, hop 256, fmax 12000)
    em.vec("spk.mel_basis", lambda: slaney_mel_basis(24000, 1024, 128, 0, 12000), [128, 513])
    return True


def emit_mimi_encoder(em: Emitter, f: STFile) -> bool:
    names = [k for k in f.keys() if k.startswith("encoder.")]
    if not names:
        return False
    done = set()
    for name in names:
        key = name[len("encoder."):]
        shape = f.shape(name)
        if key.endswith(".initialized"):
            continue
        m = re.search(r"^(quantizer\..*\.layers\.\d+\.codebook)\.(cluster_usage|embed_sum)$", key)
        if m:
            base = m.group(1)
            if base in done:
                continue
            done.add(base)
            es, cu = "encoder." + base + ".embed_sum", "encoder." + base + ".cluster_usage"
            em.mat("enc." + base + ".embed", lambda es=es, cu=cu: f.get(es) / np.maximum(f.get(cu), 1e-5)[:, None],
                   f.shape(es), dtype="f32")
            continue
        out = "enc." + key
        if len(shape) == 3:
            em.mat(out, lambda n=name: _conv_kmajor(f.get(n)), [shape[0], shape[1] * shape[2]], dtype="f32")
        elif len(shape) == 2:
            em.mat(out, lambda n=name: f.get(n), shape, dtype="f32")
        else:
            em.vec(out, lambda n=name: f.get(n), shape)
    return True


MIMI_KEYS = ("sampling_rate", "audio_channels", "hidden_size", "num_filters", "num_residual_layers", "upsampling_ratios",
             "kernel_size", "last_kernel_size", "residual_kernel_size", "dilation_growth_rate", "use_causal_conv",
             "pad_mode", "compress", "codebook_size", "codebook_dim", "num_quantizers", "use_conv_shortcut",
             "vector_quantization_hidden_dimension", "num_semantic_quantizers", "num_hidden_layers", "intermediate_size",
             "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act", "norm_eps", "rope_theta",
             "frame_rate")
MIMI_DEFAULTS = dict(sampling_rate=24000, audio_channels=1, hidden_size=512, num_filters=64, num_residual_layers=1,
                     upsampling_ratios=[8, 6, 5, 4], kernel_size=7, last_kernel_size=3, residual_kernel_size=3,
                     dilation_growth_rate=2, use_causal_conv=True, pad_mode="constant", compress=2, codebook_size=2048,
                     codebook_dim=256, num_quantizers=32, use_conv_shortcut=False,
                     vector_quantization_hidden_dimension=256, num_semantic_quantizers=1, num_hidden_layers=8,
                     intermediate_size=2048, num_attention_heads=8, num_key_value_heads=8, head_dim=None,
                     hidden_act="gelu", norm_eps=1e-5, rope_theta=10000.0, frame_rate=12.5)
SPK_DEFAULTS = dict(mel_dim=128, enc_dim=1024, enc_channels=[512, 512, 512, 512, 1536], enc_kernel_sizes=[5, 3, 3, 3, 1],
                    enc_dilations=[1, 2, 3, 4, 1], enc_attention_channels=128, enc_res2net_scale=8, enc_se_channels=128,
                    sample_rate=24000)


# ---------------- токенизатор ----------------
def tokenizer_blobs(d: Path):
    vocab = json.load(open(d / "vocab.json", encoding="utf-8"))
    toks = {int(i): t for t, i in vocab.items()}
    specials = []
    tcfg = d / "tokenizer_config.json"
    if tcfg.exists():
        for i, v in json.load(open(tcfg, encoding="utf-8")).get("added_tokens_decoder", {}).items():
            toks[int(i)] = v["content"]
            if v.get("special", False):
                specials.append(v["content"])
    merges = [l.rstrip("\n") for l in open(d / "merges.txt", encoding="utf-8") if l.strip() and not l.startswith("#version")]
    vtxt = "\n".join(f"{i}\t{toks[i]}" for i in sorted(toks)).encode()
    mtxt = "\n".join(merges).encode()
    return ({"tokenizer.vocab": np.frombuffer(vtxt, np.uint8), "tokenizer.merges": np.frombuffer(mtxt, np.uint8)}, specials)


def merged(defaults: dict, given: dict) -> dict:
    out = dict(defaults)
    out.update({k: v for k, v in (given or {}).items() if v is not None})
    return out


def build(model_dir: str, out: str, dtype: str = "bf16", quant: str = "", codec_dtype: str = "", sha: bool = True):
    d = Path(model_dir)
    cfg = json.load(open(d / "config.json"))
    tcfg = cfg.get("talker_config", {})
    pcfg = tcfg.get("code_predictor_config", {})
    tc = merged(TALKER_DEFAULTS, {k: v for k, v in tcfg.items() if k != "code_predictor_config"})
    pc = merged(PRED_DEFAULTS, pcfg)
    pc["num_code_groups"] = pc.get("num_code_groups", tc["num_code_groups"])
    if not tc.get("head_dim"):
        tc["head_dim"] = tc["hidden_size"] // tc["num_attention_heads"]
    tc["codec_language_id"] = tcfg.get("codec_language_id") or {}
    st_dir = d / "speech_tokenizer"
    scfg = json.load(open(st_dir / "config.json"))
    cc = merged(CODEC_DEFAULTS, scfg.get("decoder_config", {}))
    if "codebook_dim" not in cc:
        raise SystemExit("speech_tokenizer/config.json: decoder_config.codebook_dim is missing")
    ids = merged(IDS_DEFAULTS, cfg)
    ids = {"im_start": ids["im_start_token_id"], "im_end": ids["im_end_token_id"], "tts_pad": ids["tts_pad_token_id"],
           "tts_bos": ids["tts_bos_token_id"], "tts_eos": ids["tts_eos_token_id"]}

    em = Emitter(dtype)
    ck = open_checkpoint(d)
    emit_main(em, ck, quant, tc, pc)
    st_file = STFile(st_dir / "model.safetensors")
    emit_codec(em, st_file, (codec_dtype or "f32"), cc)
    voice = {}
    if emit_speaker_encoder(em, ck):
        voice["speaker_encoder"] = merged(SPK_DEFAULTS, cfg.get("speaker_encoder_config", {}))
    if emit_mimi_encoder(em, st_file):
        ec = merged(MIMI_DEFAULTS, {k: v for k, v in scfg.get("encoder_config", {}).items() if k in MIMI_KEYS})
        if not ec.get("head_dim"):
            ec["head_dim"] = ec["hidden_size"] // ec["num_attention_heads"]
        ec["valid_quantizers"] = scfg.get("encoder_valid_num_quantizers", tc["num_code_groups"])
        ec["downsample_rate"] = scfg.get("encode_downsample_rate", 1920)
        voice["encoder"] = ec
    tok_blobs, specials = tokenizer_blobs(d)

    gen = {}
    gp = d / "generate_config.json"
    g = json.load(open(gp)) if gp.exists() else {}
    for src, dst in (("temperature", "temperature"), ("top_k", "top_k"), ("top_p", "top_p"),
                     ("repetition_penalty", "repetition_penalty"), ("subtalker_temperature", "sub_temperature"),
                     ("subtalker_top_k", "sub_top_k"), ("subtalker_top_p", "sub_top_p"), ("max_new_tokens", "max_frames")):
        if src in g:
            gen[dst] = g[src]
    gen.setdefault("chunk_frames", 4)
    spf = int(np.prod(cc["upsample_rates"]) * np.prod(cc["upsampling_ratios"]))
    variant = "-".join(str(cfg.get(k, "")) for k in ("tokenizer_type", "tts_model_size", "tts_model_type") if cfg.get(k)).lower() \
        or d.name
    manifest = {
        "format": "1.0",
        "model": {"family": "qwen3-tts", "variant": variant, "type": cfg.get("tts_model_type", ""),
                  "weights_sha256": weights_sha256(model_dir) if sha else ""},
        "target": {"backend": "cpu", "arch": "any", "features": []},
        "audio": {"sample_rate": scfg.get("output_sample_rate", 24000), "samples_per_frame": spf,
                  "frame_rate": scfg.get("output_sample_rate", 24000) / spf, "num_codebooks": tc["num_code_groups"],
                  "codebook_size": pc["vocab_size"]},
        "stages": [{"name": "talker", "kind": "autoregressive"}, {"name": "code_predictor", "kind": "subloop"},
                   {"name": "decoder", "kind": "streaming_conv", "context": "left_only"}],
        "precision": {"weights": dtype, "quant": quant or "none", "codec": codec_dtype or "f32"},
        "generation": gen,
        "tokenizer": {"specials": specials},
        "qwen3": {"talker": tc, "predictor": pc, "codec": cc, "ids": ids, **voice},
    }
    blobs = dict(em.blobs)
    blobs.update(tok_blobs)
    write_tmod(out, manifest, blobs)
    mb = os.path.getsize(out) / 2**20
    print(f"{out}: {len(blobs)} blobs, {mb:.1f} MiB", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="папка чекпойнта (или repo id на Hugging Face)")
    ap.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"])
    ap.add_argument("--codec-dtype", default="", choices=["", "f32", "f16", "bf16"], help="dtype весов декодера кодека; по умолчанию f32 (bf16 даёт ошибку аудио ~5.8e-2)")
    ap.add_argument("--quant", default="", choices=["", "q8", "q4", "q4mlp", "q4full"])
    ap.add_argument("--no-hash", action="store_true", help="не считать sha256 весов (быстрее, но голоса не узнают модель)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    model = a.model
    if not os.path.isdir(model):
        from huggingface_hub import snapshot_download
        model = snapshot_download(model)
    if any(Path(model).glob("cosyvoice2*.yaml")):   # CosyVoice2 release folder
        from .build_cosyvoice import build as build_cosy
        build_cosy(model, a.out, a.dtype, "f32", True, sha=not a.no_hash, quant="q8" if a.quant == "q8" else "")
        return
    if (Path(model) / "t3_cfg.safetensors").exists() and (Path(model) / "s3gen.safetensors").exists():   # Chatterbox
        from .build_chatterbox import build as build_cb
        build_cb(model, a.out, a.dtype, "f32", True, sha=not a.no_hash, quant="q8" if a.quant == "q8" else "")
        return
    build(model, a.out, a.dtype, a.quant, a.codec_dtype, sha=not a.no_hash)


if __name__ == "__main__":
    main()
