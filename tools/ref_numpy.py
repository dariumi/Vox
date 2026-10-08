#!/usr/bin/env python3
"""Независимая numpy-реализация инференса Qwen3-TTS, транскрибированная из кода оригинала
(qwen_tts/core/models/modeling_qwen3_tts.py и tokenizer_12hz): без KV-кэша (каждый шаг пересчитывает
всю последовательность), без стриминга (декодер кодека — одним проходом по ref+gen кодам, как в
generate_voice_clone), жадное декодирование. Пишет дамп в том же формате, что tools/dump_reference.py,
и проверяется `vox verify`.

    python tools/ref_numpy.py --model /tmp/tiny --voice v.tvoice --text "Привет" --lang ru --out case.safetensors
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from vox.build import STFile, open_checkpoint  # noqa: E402
from vox.formats import VoiceProfile, save_safetensors  # noqa: E402

import json  # noqa: E402
from pathlib import Path  # noqa: E402

f32 = np.float32
_erf = np.frompyfunc(math.erf, 1, 1)


def erf(x):
    return _erf(x.astype(np.float64)).astype(f32)


def gelu(x):
    return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(f32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(f32)


def rmsnorm(x, w, eps):
    x = x.astype(f32)
    var = (x ** 2).mean(-1, keepdims=True)
    return (w * (x / np.sqrt(var + eps))).astype(f32)


def linear(x, W, b=None):
    y = x @ W.T
    return (y + b if b is not None else y).astype(f32)


def rope_cos_sin(L, hd, theta):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.int64).astype(f32) / hd))
    freqs = np.outer(np.arange(L, dtype=f32), inv.astype(f32)).astype(f32)
    emb = np.concatenate([freqs, freqs], -1)
    return np.cos(emb).astype(f32), np.sin(emb).astype(f32)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


class Ref:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.cfg = json.load(open(d / "config.json"))
        self.tc = self.cfg["talker_config"]
        self.pc = self.tc["code_predictor_config"]
        sc = json.load(open(d / "speech_tokenizer" / "config.json"))
        self.cc = sc["decoder_config"]
        ck = open_checkpoint(d)
        self.W = {k: f.get(k) for k, f in ck.items()}
        cf = STFile(d / "speech_tokenizer" / "model.safetensors")
        self.C = {k[len("decoder."):]: cf.get(k) for k in cf.keys() if k.startswith("decoder.")}
        from test_tokenizer import PAT  # noqa: F401
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex, AddedToken
        vocab = json.load(open(d / "vocab.json", encoding="utf-8"))
        merges = [tuple(l.rstrip("\n").split(" ", 1)) for l in open(d / "merges.txt", encoding="utf-8")
                  if l.strip() and not l.startswith("#version")]
        tk = Tokenizer(models.BPE(vocab, merges))
        tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT), "isolated", False),
                                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tk.add_special_tokens([AddedToken(t, special=True) for t in ("<|endoftext|>", "<|im_start|>", "<|im_end|>")])
        self.tok = tk

    # ---------- общий трансформер (как Qwen3TTSDecoderLayer / DecoderTransformerLayer) ----------
    def transformer(self, x, prefix, W, cfg, qk_norm, window=None, layer_scale=False):
        L, H = x.shape
        nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = cfg.get("head_dim") or H // nh
        eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
        cos, sin = rope_cos_sin(L, hd, theta)
        mask = np.tril(np.ones((L, L), bool))
        if window:
            mask &= (np.arange(L)[:, None] - np.arange(L)[None, :]) < window
        for i in range(cfg["num_hidden_layers"]):
            p = f"{prefix}layers.{i}."
            h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)
            q = linear(h, W[p + "self_attn.q_proj.weight"]).reshape(L, nh, hd)
            k = linear(h, W[p + "self_attn.k_proj.weight"]).reshape(L, nkv, hd)
            v = linear(h, W[p + "self_attn.v_proj.weight"]).reshape(L, nkv, hd)
            if qk_norm:
                q = rmsnorm(q, W[p + "self_attn.q_norm.weight"], eps)
                k = rmsnorm(k, W[p + "self_attn.k_norm.weight"], eps)
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
            out = np.zeros((L, nh, hd), f32)
            for hh in range(nh):
                g = hh // (nh // nkv)
                s = (q[:, hh, :] @ k[:, g, :].T) * hd ** -0.5
                s = np.where(mask, s, -np.inf)
                s = np.exp(s - s.max(-1, keepdims=True))
                s = s / s.sum(-1, keepdims=True)
                out[:, hh, :] = s @ v[:, g, :]
            o = linear(out.reshape(L, nh * hd), W[p + "self_attn.o_proj.weight"])
            if layer_scale:
                o = W[p + "self_attn_layer_scale.scale"] * o
            x = x + o
            h = rmsnorm(x, W[p + "post_attention_layernorm.weight"], eps)
            m = linear(silu(linear(h, W[p + "mlp.gate_proj.weight"])) * linear(h, W[p + "mlp.up_proj.weight"]),
                       W[p + "mlp.down_proj.weight"])
            if layer_scale:
                m = W[p + "mlp_layer_scale.scale"] * m
            x = x + m
        return x

    # ---------- эмбеддинги ----------
    def codec_emb(self, ids):
        return self.W["talker.model.codec_embedding.weight"][np.asarray(ids)]

    def text_proj(self, ids):
        e = self.W["talker.model.text_embedding.weight"][np.asarray(ids)]
        h = silu(linear(e, self.W["talker.text_projection.linear_fc1.weight"], self.W["talker.text_projection.linear_fc1.bias"]))
        return linear(h, self.W["talker.text_projection.linear_fc2.weight"], self.W["talker.text_projection.linear_fc2.bias"])

    def pred_emb(self, j, ids):
        return self.W[f"talker.code_predictor.model.codec_embedding.{j}.weight"][np.asarray(ids)]

    # ---------- подготовка префикса (generate(): режим non_streaming_mode=False) ----------
    def prompt(self, text, language, voice: VoiceProfile):
        tc = self.tc
        ids = self.cfg
        input_id = np.array(self.tok.encode(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
                                            add_special_tokens=False).ids)
        tts_bos, tts_eos, tts_pad = self.text_proj([ids["tts_bos_token_id"], ids["tts_eos_token_id"], ids["tts_pad_token_id"]])
        lang = language.lower()
        lang_id = None if lang == "auto" else tc["codec_language_id"][lang]
        if lang_id is None:
            pre = [tc["codec_nothink_id"], tc["codec_think_bos_id"], tc["codec_think_eos_id"]]
        else:
            pre = [tc["codec_think_id"], tc["codec_think_bos_id"], lang_id, tc["codec_think_eos_id"]]
        e0 = self.codec_emb(pre)
        e1 = self.codec_emb([tc["codec_pad_id"], tc["codec_bos_id"]])
        spk = voice.speaker_emb.astype(f32).reshape(1, -1)
        cie = np.concatenate([e0, spk, e1], 0)
        role = self.text_proj(input_id[:3])
        body = np.concatenate([np.repeat(tts_pad[None], len(cie) - 2, 0), tts_bos[None]], 0) + cie[:-1]
        emb = np.concatenate([role, body], 0)
        icl = voice.mode == "icl"
        if icl:
            ref_id = np.array(voice.ref_text_tokens)[3:-2]
            text_id = input_id[3:-5]
            text_embed = np.concatenate([self.text_proj(np.concatenate([ref_id, text_id])), tts_eos[None]], 0)
            rc = voice.ref_codes.astype(np.int64)
            codec = self.codec_emb(rc[:, 0])
            for q in range(1, rc.shape[1]):
                codec = codec + self.pred_emb(q - 1, rc[:, q])
            codec_embed = np.concatenate([self.codec_emb([tc["codec_bos_id"]]), codec], 0)
            tl, cl = len(text_embed), len(codec_embed)
            if tl > cl:
                icl_in, trailing = text_embed[:cl] + codec_embed, text_embed[cl:]
            else:
                text_embed = np.concatenate([text_embed, np.repeat(tts_pad[None], cl - tl, 0)], 0)
                icl_in, trailing = text_embed + codec_embed, tts_pad[None]
            emb = np.concatenate([emb, icl_in], 0)
        else:
            first = self.text_proj(input_id[3:4]) + cie[-1:]
            emb = np.concatenate([emb, first], 0)
            trailing = np.concatenate([self.text_proj(input_id[4:-5]), tts_eos[None]], 0)
        return emb.astype(f32), trailing.astype(f32), tts_pad.astype(f32)

    # ---------- генерация ----------
    def sample(self, logits, history, rep_pen, suppress, eos, step):
        l = logits.astype(f32).copy()
        for t in set(history):  # repetition penalty (HF)
            l[t] = l[t] * rep_pen if l[t] < 0 else l[t] / rep_pen
        if step < 2:
            l[eos] = -np.inf
        l[suppress] = -np.inf
        return int(np.argmax(l))

    def generate(self, text, language, voice, max_frames, rep_pen=1.05, forced=None):
        tc, pc = self.tc, self.pc
        Q = tc["num_code_groups"]
        emb, trailing, tts_pad = self.prompt(text, language, voice)
        seq = emb
        V = tc["vocab_size"]
        suppress = [i for i in range(V - 1024, V) if i != tc["codec_eos_token_id"]]
        suppress = np.array([i for i in suppress if i >= 0])
        tl, pl, codes, hist = [], [], [], []
        for step in range(max_frames):
            h = self.transformer(seq, "talker.model.", self.W, tc, True)
            h = rmsnorm(h, self.W["talker.model.norm.weight"], tc["rms_norm_eps"])
            past = h[-1]
            logits = linear(past, self.W["talker.codec_head.weight"])
            tl.append(logits)
            code0 = self.sample(logits, hist, rep_pen, suppress, tc["codec_eos_token_id"], step)
            # code predictor (всегда выполняется, даже для кадра eos — как в HF)
            proj = lambda x: linear(x, self.W["talker.code_predictor.small_to_mtp_projection.weight"],
                                    self.W["talker.code_predictor.small_to_mtp_projection.bias"])
            inp = proj(np.stack([past, self.codec_emb([code0])[0]]))
            frame, plog = [code0], []
            for cb in range(1, Q):
                ph = self.transformer(inp, "talker.code_predictor.model.", self.W, pc, True)
                ph = rmsnorm(ph, self.W["talker.code_predictor.model.norm.weight"], pc["rms_norm_eps"])
                lg = linear(ph[-1], self.W[f"talker.code_predictor.lm_head.{cb - 1}.weight"])
                plog.append(lg)
                c = int(np.argmax(lg))
                frame.append(c)
                inp = np.concatenate([inp, proj(self.pred_emb(cb - 1, [c]))], 0)
            pl.append(np.stack(plog))
            if code0 == tc["codec_eos_token_id"]:
                break
            hist.append(code0)
            codes.append(frame)
            nxt = self.codec_emb([code0])[0]
            for q in range(1, Q):
                nxt = nxt + self.pred_emb(q - 1, [frame[q]])[0]
            nxt = nxt + (trailing[step] if step < len(trailing) else tts_pad)
            seq = np.concatenate([seq, nxt[None]], 0)
        return (np.stack(codes).astype(np.int32) if codes else np.zeros((0, Q), np.int32),
                np.stack(tl).astype(f32), np.stack(pl).astype(f32))

    # ---------- декодер кодека (tokenizer_12hz.Qwen3TTSTokenizerV2Decoder.forward) ----------
    def decode(self, codes):
        C, cfg = self.C, self.cc
        T, Q = codes.shape
        tab = lambda base: C[base + "._codebook.embedding_sum"] / np.maximum(C[base + "._codebook.cluster_usage"], 1e-5)[:, None]
        s1 = tab("quantizer.rvq_first.vq.layers.0")[codes[:, 0]]
        s2 = sum(tab(f"quantizer.rvq_rest.vq.layers.{q - 1}")[codes[:, q]] for q in range(1, Q))
        z = s1 @ C["quantizer.rvq_first.output_proj.weight"][:, :, 0].T + s2 @ C["quantizer.rvq_rest.output_proj.weight"][:, :, 0].T
        x = self.cconv(z, "pre_conv.conv")
        x = linear(x, C["pre_transformer.input_proj.weight"], C["pre_transformer.input_proj.bias"])
        pt = dict(cfg, rms_norm_eps=cfg["rms_norm_eps"])
        x = self.transformer(x, "pre_transformer.", C, pt, False, window=cfg["sliding_window"], layer_scale=True)
        x = rmsnorm(x, C["pre_transformer.norm.weight"], cfg["rms_norm_eps"])
        x = linear(x, C["pre_transformer.output_proj.weight"], C["pre_transformer.output_proj.bias"])
        for s in range(len(cfg["upsampling_ratios"])):
            p = f"upsample.{s}."
            x = self.transconv(x, p + "0.conv")
            res = x
            y = self.cconv(x, p + "1.dwconv.conv", groups=x.shape[1])
            y = (y - y.mean(-1, keepdims=True)) / np.sqrt(y.var(-1, keepdims=True) + 1e-6)
            y = y * C[p + "1.norm.weight"] + C[p + "1.norm.bias"]
            y = gelu(linear(y, C[p + "1.pwconv1.weight"], C[p + "1.pwconv1.bias"]))
            y = linear(y, C[p + "1.pwconv2.weight"], C[p + "1.pwconv2.bias"])
            x = res + C[p + "1.gamma"] * y
        x = self.cconv(x, "decoder.0.conv")
        nb = len(cfg["upsample_rates"])
        for b in range(nb):
            p = f"decoder.{b + 1}.block."
            x = self.snake(x, p + "0")
            x = self.transconv(x, p + "1.conv")
            for u, dil in enumerate((1, 3, 9)):
                q = p + f"{u + 2}."
                r = x
                y = self.snake(x, q + "act1")
                y = self.cconv(y, q + "conv1.conv", dil=dil)
                y = self.snake(y, q + "act2")
                y = self.cconv(y, q + "conv2.conv")
                x = y + r
        x = self.snake(x, f"decoder.{nb + 1}")
        x = self.cconv(x, f"decoder.{nb + 2}.conv")
        return np.clip(x[:, 0], -1, 1).astype(f32)

    def snake(self, x, p):
        a, b = np.exp(self.C[p + ".alpha"]), np.exp(self.C[p + ".beta"])
        return (x + (1.0 / (b + 1e-9)) * np.sin(x * a) ** 2).astype(f32)

    def cconv(self, x, p, dil=1, groups=1):
        W, b = self.C[p + ".weight"], self.C[p + ".bias"]  # [cout, cin/groups, K]
        co, cig, K = W.shape
        pad = (K - 1) * dil
        xp = np.concatenate([np.zeros((pad, x.shape[1]), f32), x], 0)
        T = x.shape[0]
        out = np.zeros((T, co), f32)
        for k in range(K):
            seg = xp[k * dil:k * dil + T]
            if groups == 1:
                out += seg @ W[:, :, k].T
            else:
                out += seg * W[:, 0, k][None, :]
        return (out + b).astype(f32)

    def transconv(self, x, p):
        W, b = self.C[p + ".weight"], self.C[p + ".bias"]  # [cin, cout, K]; stride = K для апсэмплинга, K/2 для блоков
        cin, co, K = W.shape
        stride = K if p.startswith("upsample") else K // 2
        T = x.shape[0]
        y = np.zeros(((T - 1) * stride + K, co), f32)
        for t in range(T):
            y[t * stride:t * stride + K] += np.einsum("i,iok->ko", x[t], W).astype(f32)
        y = y[:y.shape[0] - (K - stride)]
        return (y + b).astype(f32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = Ref(a.model)
    voice = VoiceProfile.load(a.voice)
    codes, tl, pl = ref.generate(a.text, a.lang, voice, a.max_frames)
    ref_frames = 0 if voice.mode != "icl" else voice.ref_codes.shape[0]
    allc = codes if ref_frames == 0 else np.concatenate([voice.ref_codes.astype(np.int64), codes], 0)
    wav = ref.decode(allc.astype(np.int64))
    spf = len(wav) // len(allc)
    wav = wav[ref_frames * spf:]
    save_safetensors(a.out, {"wav": wav, "talker_logits": tl, "predictor_logits": pl, "codes": codes},
                     {"text": a.text, "language": a.lang, "ref_frames": str(ref_frames), "voice": os.path.abspath(a.voice),
                      "generator": "ref_numpy"})
    print(f"{a.out}: {len(codes)} frames, {len(tl)} steps, {len(wav)} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
